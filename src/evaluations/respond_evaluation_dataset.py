import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from transformers.pipelines.base import Pipeline

sys.path.append(Path(__file__).parent.parent.name)
from generate_star_wars_prompts import EvaluationEntry

from utils.torch_random import set_seed

device_map = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = os.getenv("HF_TOKEN")
set_seed(42)


####################################################################
# Part 0: Parse arguments
####################################################################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_id", type=str, required=True)
    parser.add_argument("--reinforced_model_id", type=str, required=True)
    parser.add_argument("--generic_model_id", type=str, required=True)
    parser.add_argument("--evaluation_dataset_folder", type=str, required=True)
    args = parser.parse_args()

    return args


####################################################################
# Part 1: Load a model and its tokenizer (generic-finetuned, finetunded, baseline)
####################################################################
def load_model(
    base_model_id: str,
    quantized_model_id: str,
    quantized: bool = False,
    task: str = "text-generation",
) -> Pipeline:
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True,
    # )

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
        # quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    if quantized:
        model = PeftModel.from_pretrained(
            model=base_model,
            model_id=quantized_model_id,
            adapter_name="lora_1",
            is_trainable=True,
        ).to(device_map)
        model.merge_and_unload()
    else:
        model = base_model

    model.eval()  # set model to evaluation mode

    pipe = pipeline(
        task=task,
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    return pipe


# def load_multi_models(
#     model_ids: Dict[str, str], task: str = "text-generation"
# ) -> Dict[str, Pipeline]:
#     pipelines = dict()
#     for model_name, model_id in model_ids.item():
#         pipelines[model_name] = load_model(model_id, task)

#     return pipelines


####################################################################
# Part 2: Load evaluation dataset (initially without model_response or judgement)
####################################################################
def load_evaluation_dataset(dataset_folder: Path) -> List[EvaluationEntry]:
    responded_dataset = dataset_folder / "evaluation_complete_sentence_responded.json"
    unresponded_dataset = dataset_folder / "evaluation_complete_sentence.json"

    dataset_path = None
    if responded_dataset.exists():
        dataset_path = responded_dataset
    else:
        dataset_path = unresponded_dataset

    with open(dataset_path, "r") as f:
        evaluation_dataset = json.load(f)

    return evaluation_dataset


####################################################################
# Part 3: Generate response for each mdoel
####################################################################
def generate_response(
    pipe: Pipeline,
    evaluation_dataset: List[EvaluationEntry],
    model_name: str,
) -> List[EvaluationEntry]:
    responped_evaluation_dataset = list()

    for item in tqdm(evaluation_dataset):
        input_prompt = item["prompt"]
        full_prompt = f"{input_prompt}"
        output = pipe(
            full_prompt,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
        )
        generated_text = output[0]["generated_text"]

        cleaned_response = remove_prompt_from_generation(full_prompt, generated_text)

        # compose the response into model_response
        model_response: Dict[str, str] = item.get("model_response") or dict()
        model_response[model_name] = cleaned_response

        responped_evaluation_dataset.append(
            {
                "evaluation_id": item["evaluation_id"],
                "category": item["category"],
                "prompt": item["prompt"],
                "reference": item["reference"],
                "model_response": model_response,
                "judgement": None,
            }
        )

    return responped_evaluation_dataset


def remove_prompt_from_generation(prompt: str, generation: str) -> str:
    if generation.startswith(prompt):
        return generation[len(prompt) :].strip()
    return generation.strip()


def save_responded(
    evaluation_responded: List[EvaluationEntry], output_folder: str
) -> None:
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    output_file = output_folder / "evaluation_complete_sentence_responded.json"

    with open(output_file, "w") as f:
        f.write(json.dumps(evaluation_responded, ensure_ascii=False, indent=4))


def main():
    args = parse_args()

    model_ids = {
        "baseline": (args.base_model_id, None),
        "reinforced": (args.base_model_id, args.reinforced_model_id),
        "generic": (args.base_model_id, args.generic_model_id),
    }

    for model_name, (base_model_id, quantized_model_id) in model_ids.items():
        print(f"Responding the evaluation dataset with {model_name} model...")

        quantized = True if quantized_model_id else False

        # 1. Load model (baseline, reinforced, generic)
        # pipelines = load_multi_models(model_ids)
        pipe = load_model(
            base_model_id=base_model_id,
            quantized_model_id=quantized_model_id,
            quantized=quantized,
            task="text-generation",
        )

        # 2. Load evaluation dataset
        # evaluation_dataset_folder = "data/evaluation/llm-as-a-judge"
        evaluation_dataset = load_evaluation_dataset(
            Path(args.evaluation_dataset_folder)
        )

        # 3. Generate response with the fine-tuned model
        evaluation_responded = generate_response(pipe, evaluation_dataset, model_name)

        # 4. Save the responded evaluation dataset
        save_responded(evaluation_responded, args.evaluation_dataset_folder)

        del pipe
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
