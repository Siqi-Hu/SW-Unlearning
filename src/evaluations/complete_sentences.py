import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
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
def load_model_with_adapter(
    base_model_id: str,
    lora_adapter_id: str | None,
    task: str = "text-generation",
    return_pipe: bool = True,
) -> Pipeline | Tuple[PreTrainedModel, PreTrainedTokenizer]:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

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
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    if lora_adapter_id:
        model = PeftModel.from_pretrained(
            model=base_model,
            model_id=lora_adapter_id,
            adapter_name="lora_1",
            is_trainable=True,
        ).to(device_map)
        model.merge_and_unload()
    else:
        model = base_model

    model.eval()  # set model to evaluation mode

    if return_pipe:
        pipe = pipeline(
            task=task,
            model=model,
            tokenizer=tokenizer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        return pipe

    else:
        return model, tokenizer


# def load_model(model_id: str, task: str = "text-generation") -> pipeline:
#     """
#     Loads a fully merged (non-LoRA, non-quantized) model and tokenizer from model_id.
#     Assumes model was saved via `model.save_pretrained(...)` and merged from any LoRA adapters.
#     """
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True,
#     )

#     tokenizer = AutoTokenizer.from_pretrained(
#         model_id,
#         trust_remote_code=True,
#         padding_side="right",
#         truncation=True,  # Ensures sequences longer than 512 are truncated
#         max_length=512,  # Ensures no input exceeds 512 tokens
#     )

#     # Make sure we have a pad token
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         device_map="auto",
#         trust_remote_code=True,
#         quantization_config=bnb_config,
#     )

#     model.eval()

#     pipe = pipeline(
#         task=task,
#         model=model,
#         tokenizer=tokenizer,
#         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
#         eos_token_id=tokenizer.eos_token_id,
#     )

#     return pipe


def load_model_tokenizer(
    model_id: str,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="right",
        truncation=True,  # Ensures sequences longer than 512 are truncated
        max_length=512,  # Ensures no input exceeds 512 tokens
    )

    # Make sure we have a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
    )

    return model, tokenizer


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
        full_prompt = input_prompt  # possibly add some other context to the prompt
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
        cut_response = cut_after_first_full_stop(cleaned_response)

        print(f"Model: {model_name}")
        print(f"Prompt: {full_prompt}")
        print(f"Generated Response: {generated_text}")
        print(f"Cleaned Response: {cleaned_response}")
        print(f"Cut Response: {cut_response}")
        print(
            f"Reference: {remove_prompt_from_generation(item['prompt'], item['reference'])}"
        )
        print("================================")
    #     # compose the response into model_response
    #     model_response: Dict[str, str] = item.get("model_response") or dict()
    #     model_response[model_name] = cleaned_response

    #     responped_evaluation_dataset.append(
    #         {
    #             "evaluation_id": item["evaluation_id"],
    #             "category": item["category"],
    #             "prompt": item["prompt"],
    #             "reference": item["reference"],
    #             "model_response": model_response,
    #             "judgement": None,
    #         }
    #     )

    # return responped_evaluation_dataset


def remove_prompt_from_generation(prompt: str, generation: str) -> str:
    if generation.startswith(prompt):
        return generation[len(prompt) :].strip()
    return generation.strip()


def cut_after_first_full_stop(response: str) -> str:
    index = response.find(".")
    return response[: index + 1] if index != -1 else response


def next_token_prob(model, tokenizer, input_text, top_k=10):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    next_token_logits = logits[0, -1, :]
    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)

    top_k_probs, top_k_ids = torch.topk(probs, top_k)
    top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_ids.tolist())

    return list(zip(top_k_tokens, top_k_probs.tolist()))


def next_token_probs(model_name, model, tokenizer, evaluation_dataset, top_k=10):
    for item in tqdm(evaluation_dataset):
        input_prompt = item["prompt"]

        results = next_token_prob(model, tokenizer, input_prompt, top_k)

        print(f"Model: {model_name}")
        print(f"Prompt: {input_prompt}")
        print(f"Results: {results}")
        print("================================")


def main():
    args = parse_args()

    model_ids = {
        "baseline": (args.base_model_id, None),
        "reinforced": (args.base_model_id, args.reinforced_model_id),
        "generic": (
            args.base_model_id,
            args.generic_model_id,
        ),
    }

    for model_name, (base_model_id, lora_adapter_id) in model_ids.items():
        print(f"Responding the evaluation dataset with {model_name} model...")

        # 1. Load model (baseline, reinforced, generic)
        pipe = load_model_with_adapter(
            base_model_id=base_model_id,
            lora_adapter_id=lora_adapter_id,
            task="text-generation",
            return_pipe=True,
        )

        # loading model and tokenizer for next token probability
        model, tokenizer = load_model_with_adapter(
            base_model_id=base_model_id,
            lora_adapter_id=lora_adapter_id,
            task="text-generation",
            return_pipe=False,
        )

        # 2. Load evaluation dataset
        evaluation_dataset = load_evaluation_dataset(
            Path(args.evaluation_dataset_folder)
        )

        # # 3. Generate response with the fine-tuned model
        # evaluation_responded = generate_response(pipe, evaluation_dataset, model_name)

        next_token_probs(model_name, model, tokenizer, evaluation_dataset, 10)


if __name__ == "__main__":
    main()
