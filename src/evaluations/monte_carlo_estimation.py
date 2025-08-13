import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from jinja2 import Template
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    pipeline,
)
from transformers.pipelines.base import Pipeline

sys.path.append(Path(__file__).parent.parent.name)
from utils.torch_random import set_seed

DEVICE_MAP = "cuda" if torch.cuda.is_available() else "cpu"
# device_map = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = os.getenv("HF_TOKEN")
set_seed(42)


def load_prompts(dataset_folder: Path) -> List[str]:
    dataset_path = dataset_folder / "evaluation_complete_sentence.json"

    with open(dataset_path, "r") as f:
        evaluation_dataset = json.load(f)

    prompts = [data.get("prompt") for data in evaluation_dataset]

    return prompts


def load_model(
    base_model_id: str,
    lora_adapter_id: str,
    adapter: bool = False,
    task: str = "text-generation",
) -> Pipeline:
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
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    if adapter:
        model = PeftModel.from_pretrained(
            model=base_model,
            model_id=lora_adapter_id,
            adapter_name="lora_1",
            is_trainable=True,
        ).to(DEVICE_MAP)
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


def load_judge_pipeline(
    judge_model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        judge_model_id,
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

    model = AutoModelForCausalLM.from_pretrained(
        judge_model_id,
        torch_dtype=torch.float16 if DEVICE_MAP == "cuda" else torch.float32,
        device_map="auto",
        token=HF_TOKEN,
    )

    model.eval()

    return model, tokenizer


def remove_prompt_from_generation(prompt: str, generation: str) -> str:
    if generation.startswith(prompt):
        return generation[len(prompt) :].strip()
    return generation.strip()


def generate_response(pipe: Pipeline, input_prompt: str) -> str:
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

    return cleaned_response


def load_prompt_template() -> Template:
    template_path = "./prompt_templates/llm_as_a_judge.template"
    with open(template_path, "r") as f:
        return Template(f.read())


def create_prompt(response: str) -> str:
    prompt_template = load_prompt_template()

    prompt = prompt_template.render({"response": response})

    return prompt


def judge_response(
    response: str,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    temperature: float,
) -> str | int:
    full_prompt = create_prompt(response)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE_MAP)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=True if temperature != 0.0 else False,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
    )

    prediction = remove_prompt_from_generation(
        full_prompt, tokenizer.decode(outputs[0], skip_special_tokens=True)
    )

    return prediction


def monte_carlo_simulation(
    prompts: List[str],
    model_name: str,
    model_config: Tuple[str, str | None],
    evaluation_dataset_folder: Path,
    temperatures: List[float] = np.linspace(0.0, 2.0, num=21),
    judge_model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
) -> Dict[str, pd.DataFrame]:
    iter_id = 0
    total_iters = len(temperatures) * len(prompts)

    judge_model, judge_tokenizer = load_judge_pipeline(judge_model_id)

    base_model_id, lora_adapter_id = model_config

    model_labels = dict()

    adapter = True if lora_adapter_id else False
    pipe = load_model(
        base_model_id=base_model_id,
        lora_adapter_id=lora_adapter_id,
        adapter=adapter,
        task="text-generation",
    )

    for temperature in temperatures:
        temperature_id = f"temp_{temperature}"
        temperature_labels = list()
        for prompt_id, prompt in enumerate(prompts):
            response = generate_response(pipe, prompt)
            label = judge_response(response, judge_tokenizer, judge_model, temperature)
            temperature_labels.append(label)

            print(
                f"Progress: {iter_id:>5}/{total_iters:<5} |"
                f"Model: {model_name:<10} | "
                f"Temperature: {temperature:>5.2f} | "
                f"Prompt: {prompt_id:>3} | "
                f"Classification: {label}"
            )
            iter_id += 1

        model_labels[temperature_id] = temperature_labels

    model_labels_df = pd.DataFrame(model_labels)

    save_results(model_labels_df, evaluation_dataset_folder, model_name)

    del pipe
    torch.cuda.empty_cache()


def save_results(
    model_labels_df: pd.DataFrame, evaluation_dataset_folder: Path, model_name: str
):
    model_labels_file_name = (
        evaluation_dataset_folder / f"{model_name}_labels_over_temperature.csv"
    )
    model_labels_df.to_csv(model_labels_file_name, sep=",")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    generic_model_name = (
        "Llama2-7B-lora-r-32-generic-step-1050-lr-1e-5-labels_40.0-optimized"
    )
    evaluation_dataset_folder = Path(
        f"./data/evaluation/sentence_completion_tasks/{generic_model_name}"
    )
    prompts = load_prompts(evaluation_dataset_folder)

    base_model_id = "meta-llama/Llama-2-7b-hf"
    reinforced_model_id = "Siqi-Hu/Llama2-7B-lora-r-32-finetuned-epoch-4"
    generic_model_id = f"Siqi-Hu/{generic_model_name}"

    model_configs = {
        "baseline": (base_model_id, None),
        "reinforced": (base_model_id, reinforced_model_id),
        "generic": (base_model_id, generic_model_id),
    }

    judge_model_id = "mistralai/Mistral-7B-Instruct-v0.2"

    monte_carlo_simulation(
        prompts,
        model_name=args.model,
        model_config=model_configs.get(args.model),
        evaluation_dataset_folder=evaluation_dataset_folder,
        temperatures=np.linspace(0.0, 2.0, num=21),
        judge_model_id=judge_model_id,
    )


if __name__ == "__main__":
    main()
