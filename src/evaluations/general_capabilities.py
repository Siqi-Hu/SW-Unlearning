import os
import sys
from pathlib import Path
from random import sample
from typing import Dict, List, Tuple

import polars as pl
import torch
from datasets import Dataset, load_dataset
from jinja2 import Template
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

sys.path.append(Path(__file__).parent.parent.name)
from utils.torch_random import set_seed

DEVICE_MAP = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = os.getenv("HF_TOKEN")
set_seed(42)

DATASET_CONFIGS = {
    "mmlu": {
        "dataset_id": "cais/mmlu",
        "subsets_size": {
            # "high_school_world_history": 100,
            # "high_school_geography": 100,
            # logic & reasoning
            "formal_logic": 100,
            "logical_fallacies": 100,
            # Math & Quant
            "college_mathematics": 100,
            "high_school_statistics": 100,
            # Science
            "college_biology": 100,
            "college_physics": 100,
            # Social Science
            "high_school_psychology": 100,
            "sociology": 100,
            # Humanities
            "philosophy": 100,
            "high_school_world_history": 100,
            # Professional Knowledge
            "professional_law": 100,
            "medical_genetics": 100,
            # "abstract_algebra": 100,
            # "anatomy": 100,
            # "astronomy": 100,
            # "business_ethics": 100,
            # "clinical_knowledge": 100,
            # "college_biology": 100,
            # "college_chemistry": 100,
            # "college_computer_science": 100,
            # "college_mathematics": 100,
            # "college_medicine": 100,
            # "college_physics": 100,
            # "computer_security": 100,
            # "conceptual_physics": 100,
            # "econometrics": 100,
            # "electrical_engineering": 100,
            # "elementary_mathematics": 100,  # 300
            # "formal_logic": 100,
            # "global_facts": 100,
            # "high_school_biology": 100,
            # "high_school_chemistry": 100,
            # "high_school_computer_science": 100,
            # "high_school_european_history": 100,
            # "high_school_geography": 100,
            # "high_school_government_and_politics": 100,
            # "high_school_macroeconomics": 100,
            # "high_school_mathematics": 100,
            # "high_school_microeconomics": 100,
            # "high_school_physics": 100,
            # "high_school_psychology": 100,
            # "high_school_statistics": 100,
            # "high_school_us_history": 100,
            # "high_school_world_history": 100,
            # "human_aging": 100,
            # "human_sexuality": 100,
            # "international_law": 100,
            # "jurisprudence": 100,
            # "logical_fallacies": 100,
            # "machine_learning": 100,
            # "management": 100,
            # "marketing": 100,
            # "medical_genetics": 100,
            # "miscellaneous": 100,
            # "moral_disputes": 100,
            # "moral_scenarios": 100,
            # "nutrition": 100,
            # "philosophy": 100,
            # "prehistory": 100,
            # "professional_accounting": 100,
            # "professional_law": 100,
            # "professional_medicine": 100,
            # "professional_psychology": 100,
            # "public_relations": 100,
            # "security_studies": 100,
            # "sociology": 100,
            # "us_foreign_policy": 100,
            # "virology": 100,
            # "world_religions": 100,
        },
        "splits": ["test"],
    },
    "winogrande": {
        "dataset_id": "coref-data/winogrande_raw",
        "subsets_size": {"winogrande_m": 500},
        "splits": [
            # "train",
            "validation"
        ],
    },
}


def load_model_with_adapter(
    base_model_id: str,
    lora_adapter_id: str | None,
    sub_folder: str | None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        padding_side="right",
        truncation=True,  # Ensures sequences longer than 512 are truncated
        max_length=512,  # Ensures no input exceeds 512 tokens
        token=HF_TOKEN,
    )

    # Set pad token as eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map=DEVICE_MAP,
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    if lora_adapter_id:
        if sub_folder:
            model = PeftModel.from_pretrained(
                model=base_model,
                model_id=lora_adapter_id,
                subfolder=f"{sub_folder}/lora-final",
                adapter_name=sub_folder,
                is_trainable=True,
            ).to(DEVICE_MAP)
            model.merge_and_unload()
        else:
            model = PeftModel.from_pretrained(
                model=base_model,
                model_id=lora_adapter_id,
                is_trainable=True,
            ).to(DEVICE_MAP)
            model.merge_and_unload()
    else:
        model = base_model

    model.eval()  # set model to evaluation mode

    return model, tokenizer


def load_benchmark_dataset(dataset_id: str, subset: str, split: str, size: int):
    dataset = load_dataset(dataset_id, subset)

    return dataset[split]


def load_prompt_template(task_name) -> Template:
    template_path = f"prompt_templates/{task_name}.txt"
    with open(template_path, "r") as f:
        return Template(f.read())


def create_prompt(prompt_template, few_shot_examples, test_question, task: str):
    prompt = ""
    for shot in few_shot_examples:
        # shot_processed = shot.copy()
        # shot_processed["answer"] = DATASET_CONFIGS[task]["index_to_letter"](
        #     shot["answer"]
        # )
        prompt += prompt_template.render(**shot) + "\n\n"

    prompt += prompt_template.render(**test_question).replace("{{ answer }}", "")

    return prompt


def remove_prompt_from_generation(prompt: str, generation: str) -> str:
    if generation.startswith(prompt):
        return generation[len(prompt) :].strip()
    return generation.strip()


def run_5_shot_eval(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    model_name: str,
    df: pl.DataFrame,
    finetune_step: int | None = None,
) -> pl.DataFrame:
    benchmark_column = []
    subset_column = []
    split_column = []
    total_column = []
    correct_column = []

    for task, datasset_config in DATASET_CONFIGS.items():
        prompt_template = load_prompt_template(task)

        dataset_id = datasset_config["dataset_id"]
        subsets_size: Dict[str, int] = datasset_config["subsets_size"]
        splits = datasset_config["splits"]

        for subset, size in subsets_size.items():
            for split in splits:
                test_set = load_benchmark_dataset(
                    dataset_id=dataset_id, subset=subset, split=split, size=size
                )

                correct = 0
                for test_id in tqdm(range(size)):
                    test_example = test_set[test_id].copy()

                    test_question = test_example.copy()
                    del test_question["answer"]
                    test_answer = test_example["answer"]

                    few_shot_examples = sample(
                        [
                            example
                            for idx, example in enumerate(test_set)
                            if idx != test_id
                        ],
                        5,
                    )

                    prompt = create_prompt(
                        prompt_template, few_shot_examples, test_question, task
                    )
                    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE_MAP)
                    outputs = model.generate(**inputs, max_new_tokens=1)

                    prediction = remove_prompt_from_generation(
                        prompt, tokenizer.decode(outputs[0], skip_special_tokens=True)
                    )

                    if str(prediction) == str(test_answer):
                        correct += 1

                # print(f"Accuracy over {total} examples: {accuracy:.2%}")
                print(
                    f"benchmark: {task} | subset: {subset} | split: {split} | total: {size} | model_name: {model_name} | correct: {correct}"
                )

                benchmark_column.append(task)
                subset_column.append(subset)
                split_column.append(split)
                total_column.append(size)
                correct_column.append(correct)

    if model_name in ["reinforced", "generic"]:
        df = df.with_columns(
            pl.Series(correct_column).alias(
                f"{finetune_step}" if finetune_step else model_name
            )
        )

        print(df)

        return df
    else:
        data = {
            "benchmark": benchmark_column,
            "subset": subset_column,
            "split": split_column,
            "total": total_column,
            f"{model_name}": correct_column,
        }

        df = pl.DataFrame(data)

        print(df)

        return df


def main():
    base_model_id = "meta-llama/Llama-2-7b-hf"
    reinforced_lora_adapter_id = "Siqi-Hu/Llama2-7B-lora-r-32-finetuned-epoch-4"
    generic_lora_adapter_id = (
        "Siqi-Hu/Llama2-7B-lora-r-32-generic-step-1050-lr-1e-5-labels_40.0-optimized"
    )

    model_ids = {
        "baseline": (base_model_id, None),
        "reinforced": (base_model_id, reinforced_lora_adapter_id),
        "generic": (
            base_model_id,
            generic_lora_adapter_id,
        ),
    }

    df = None
    for model_name, (base_model_id, lora_adapter_id) in model_ids.items():
        print(f"Inference {model_name} model...")

        if model_name == "generic":
            # for checking models at each checkpoint fine-tuning step
            # for step in range(50, 1250, 50):
            #     print(f"Inference {model_name} model at checkpoint {step}...")
            #     sub_folder = f"lora-checkpoint-{step}"

            #     model, tokenizer = load_model_with_adapter(
            #         base_model_id=base_model_id,
            #         lora_adapter_id=lora_adapter_id,
            #         sub_folder=sub_folder,
            #     )

            #     df = run_5_shot_eval(tokenizer, model, model_name, df, step)
            #     del model
            #     del tokenizer

            # for optimized generic model at step 1050
            model, tokenizer = load_model_with_adapter(
                base_model_id=base_model_id,
                lora_adapter_id=lora_adapter_id,
                sub_folder=None,
            )
            df = run_5_shot_eval(tokenizer, model, model_name, df, None)

            del model
            del tokenizer

        else:
            model, tokenizer = load_model_with_adapter(
                base_model_id=base_model_id,
                lora_adapter_id=lora_adapter_id,
                sub_folder=None,
            )

            if model_name == "baseline":
                df = run_5_shot_eval(tokenizer, model, model_name, df, None)

                del model
                del tokenizer

            else:
                df = run_5_shot_eval(tokenizer, model, model_name, df, None)

                del model
                del tokenizer

    print(df)

    output_dir = (
        Path("data/evaluation/general_capabilities")
        / Path(generic_lora_adapter_id).name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path: Path = output_dir / "general_capabilities_score_2.csv"
    df.write_csv(path, separator=",")


if __name__ == "__main__":
    main()
