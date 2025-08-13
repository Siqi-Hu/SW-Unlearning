import os
import sys
from pathlib import Path
from typing import Tuple

import polars as pl
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

sys.path.append(Path(__file__).parent.parent.name)
from evaluations.generate_star_wars_prompts import EvaluationEntry
from utils.torch_random import set_seed

DEVICE_MAP = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = os.getenv("HF_TOKEN")
set_seed(42)


def load_model_with_adapter(
    base_model_id: str,
    lora_adapter_id: str | None,
    sub_folder: str | None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
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
                # adapter_name="lora_1",
                is_trainable=True,
            ).to(DEVICE_MAP)
            model.merge_and_unload()
    else:
        model = base_model

    model.eval()  # set model to evaluation mode

    return model, tokenizer


def baseline_top_k_tokens_probability(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_text: str,
    top_k: int = 10,
) -> Tuple[pl.DataFrame, torch.Tensor]:
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    next_token_logits = logits[0, -1, :]
    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)

    top_k_probs, top_k_ids = torch.topk(probs, top_k)
    top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_ids.tolist())

    data = {
        "top_k_ids": top_k_ids.tolist(),
        "top_k_tokens": top_k_tokens,
        "baseline": top_k_probs.tolist(),
    }
    df = pl.DataFrame(data)

    return df, top_k_ids


def hstack_next_token_probability(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_text: str,
    df: pl.DataFrame,
    baseline_top_k_ids: torch.Tensor,
    model_name: str,
    finetune_step: int | None,
) -> pl.DataFrame:
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    next_token_logits = logits[0, -1, :]
    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)

    baseline_top_k_probs = probs[baseline_top_k_ids]

    df = df.with_columns(
        pl.Series(baseline_top_k_probs.tolist()).alias(
            f"{finetune_step} steps" if finetune_step else model_name
        )
    )

    return df


def main():
    # llama 2
    # base_model_id = "meta-llama/Llama-2-7b-hf"
    # reinforced_lora_adapter_id = "Siqi-Hu/Llama2-7B-lora-r-32-finetuned-epoch-4"
    # generic_lora_adapter_id = (
    #     "Siqi-Hu/Llama2-7B-lora-r-32-generic-step-1200-lr-1e-5-labels_40.0-3"
    # )

    # llama 3
    base_model_id = "meta-llama/Meta-Llama-3-8B"
    reinforced_lora_adapter_id = "Siqi-Hu/Llama3-8B-lora-r-32-finetuned-epoch-3"
    generic_lora_adapter_id = (
        "Siqi-Hu/Llama3-8B-lora-r-32-generic-step-1200-lr-1e-5-labels_40.0-1"
    )

    model_ids = {
        "baseline": (base_model_id, None),
        "reinforced": (base_model_id, reinforced_lora_adapter_id),
        "generic": (
            base_model_id,
            generic_lora_adapter_id,
        ),
    }

    examples = [
        "Luke grabs for his pistol, but is hit flat in the face by a huge white claw. He falls unconscious into the snow and in a moment the terrified screams of the Tauntaun are cut short by the horrible snap of a neck being broken. The Wampa Ice Creature grabs",
        "The Wampa Ice Creature grabs",
        "Rebel spaceships, striking from a hidden base, have won their first victory against the evil",
        "RA clip is shown of Vader walking down a corridor of the",
        "During the battle, Rebel spies managed to steal secret plansto the Empire's ultimate weapon, the",
        "Pursued by the Empire's sinister agents, Princess",
        "The lovely young girl huddles in a small alcove as the stormtroopers search through the ship. She is Princess",
    ]

    for example_id, input_text in enumerate(examples):
        df = None
        baseline_top_k_ids = None
        for model_name, (base_model_id, lora_adapter_id) in model_ids.items():
            print(f"Inference {model_name} model...")

            if model_name == "generic":
                for step in range(50, 1100, 50):  # 1250  # 1100
                    print(f"Inference {model_name} model at checkpoint {step}...")
                    sub_folder = f"lora-checkpoint-{step}"

                    model, tokenizer = load_model_with_adapter(
                        base_model_id=base_model_id,
                        lora_adapter_id=lora_adapter_id,
                        sub_folder=sub_folder,
                    )

                    df = hstack_next_token_probability(
                        model,
                        tokenizer,
                        input_text=input_text,
                        df=df,
                        baseline_top_k_ids=baseline_top_k_ids,
                        model_name=model_name,
                        finetune_step=step,
                    )

                    del model
                    del tokenizer

            else:
                model, tokenizer = load_model_with_adapter(
                    base_model_id=base_model_id,
                    lora_adapter_id=lora_adapter_id,
                    sub_folder=None,
                )

                if model_name == "baseline":
                    df, baseline_top_k_ids = baseline_top_k_tokens_probability(
                        model, tokenizer, input_text=input_text, top_k=10
                    )

                    del model
                    del tokenizer

                else:
                    df = hstack_next_token_probability(
                        model,
                        tokenizer,
                        input_text=input_text,
                        df=df,
                        baseline_top_k_ids=baseline_top_k_ids,
                        model_name=model_name,
                        finetune_step=None,
                    )

                    del model
                    del tokenizer

        print(df)

        path: Path = (
            Path("data/experiment/next_token_prob_over_steps")
            / Path(generic_lora_adapter_id).name
            / f"example_{example_id}.csv"
        )
        df.write_csv(path, separator=",")


if __name__ == "__main__":
    main()
