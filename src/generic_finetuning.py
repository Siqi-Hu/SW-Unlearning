import argparse
import os
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from utils.custom_callback import MetricsSavingCallback, PushLoRACheckpointCallback

# Configuration
# dataset_name = "Siqi-Hu/Meta-Llama-3-8B-generic-predictions-starwars"
# model_name_or_path = "distilgpt2"
# output_dir = "./models/distilgpt2-starwars-generic-finetuned"
# num_train_epochs = 20
# per_device_train_batch_size = 1
# per_device_eval_batch_size = 1
# learning_rate = 5e-5
# gradient_accumulation_steps = 8
# device_map = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
HF_TOKEN = os.getenv("HF_TOKEN")


####################################################################
# Step 0: Parse arguments
####################################################################
def parse_args():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--label_column", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--hub_model_id", type=str, required=True)

    # LoRA config
    parser.add_argument("--lora_r", type=int, default=16)

    # Training config
    parser.add_argument("--max_steps", type=int, default=150)
    parser.add_argument("--num_train_epochs", type=int, default=2)

    # Optim and batch
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    args = parser.parse_args()

    return args


####################################################################
# Step 1&2: LoRA configuration + Load pretrained model and tokenizer
####################################################################
def print_trainable_parameters(model: PreTrainedModel):
    """
    Print the nuymber of trainable parameters in the model
    """
    trainable_parameters = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()

    print(
        f"trainable parameters: {trainable_parameters} || all params: {all_params} || trainable%: {100 * trainable_parameters / all_params}"
    )


def load_model_tokenizer(
    model_name: str,
    lora_r: int,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True,
    # )

    # First load the tokenizer to get vocab size
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
        truncation=True,  # Ensures sequences longer than 512 are truncated
        max_length=512,  # Ensures no input exceeds 512 tokens
        token=HF_TOKEN,
    )

    # Make sure we have a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Now load the model with the quantization config
    print("Loading model with 4-bit quantization...")
    # Fix: Using explicit device_map="cuda:0" instead of "auto"
    device_map = "auto"

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        # quantization_config=bnb_config,  # disabled as it requires libstdc++ from the system, which is too old here
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    # Set Flash Attention if available
    if hasattr(model.config, "use_flash_attention_2"):
        setattr(model.config, "use_flash_attention_2", True)
        print("Flash Attention 2 enabled")

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

    # freezing the original weights
    for name, param in model.named_parameters():
        param.requires_grad = False  # freeze the model, train adapters later
        # Only cast embeddings and layer norms to fp32 for stability
        if any(module_name in name.lower() for module_name in ["embed", "norm"]):
            if param.ndim == 1 or "embed" in name.lower():
                param.data = param.data.to(torch.float32)

    # # Unfreeze all layers except embedding & normalization:
    # for name, param in model.named_parameters():
    #     if any(module_name in name.lower() for module_name in ["embed", "norm"]):
    #         param.requires_grad = False
    #     elif param.dtype in [torch.float32, torch.float16, torch.bfloat16]:
    #         param.requires_grad = True

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)

    model.lm_head = CastOutputToFloat(model.lm_head)

    # adding lora_config to the model
    lora_config: PeftConfig = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # type of task to train on
        inference_mode=False,  # set to False for training
        r=lora_r,  # dimension of the smaller matrices (attention head)
        lora_alpha=2 * lora_r,  # scaling factor
        lora_dropout=0.1,  # dropout of LoRA layers
        # Apply LoRA to specific layers only to save memory
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config, adapter_name="lora-final")
    print_trainable_parameters(model)

    return model, tokenizer


####################################################################
# Step 3: Load Star Wars generic predictions dataset (tokens)
# train/validation dataset split
####################################################################
def load_generic_predictions_dataset(
    dataset_name: str, label_column: str = "labels_1.0"
) -> Tuple[Dataset, Dataset]:
    dataset = load_dataset(dataset_name)["train"]
    columns_to_keep = ["input_ids", label_column]
    columns_to_remove = set(dataset.column_names) - set(columns_to_keep)
    dataset = dataset.remove_columns(list(columns_to_remove))

    if label_column != "labels":
        dataset = dataset.rename_column(label_column, "labels")

    def convert_to_long(batch):
        # Convert input_ids to long tensor
        if isinstance(batch["input_ids"], list):
            batch["input_ids"] = torch.tensor(batch["input_ids"], dtype=torch.long)
        else:
            batch["input_ids"] = batch["input_ids"].long()

        # Convert labels to long tensor
        if isinstance(batch["labels"], list):
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.long)
        else:
            batch["labels"] = batch["labels"].long()
        return batch

    dataset = dataset.map(convert_to_long)

    dataset.set_format(type="torch", columns=["input_ids", "labels"])

    dataset = dataset.train_test_split(test_size=0.05)
    dataset["validation"] = dataset["test"]

    return dataset


####################################################################
# Step 4: Train and save results
####################################################################
def create_custom_data_collator(tokenizer):
    """
    Creates a simple data collator function for input-label training.
    """

    def collate_fn(batch):
        # print(f"DEBUG: Available keys in batch[0]: {list(batch[0].keys())}")

        # Verify we have the required keys
        if not all("input_ids" in item and "labels" in item for item in batch):
            raise ValueError(
                "Batch items missing required 'input_ids' or 'labels' keys"
            )

        # Get max length in batch
        max_input_len = max(len(item["input_ids"]) for item in batch)
        max_label_len = max(len(item["labels"]) for item in batch)
        max_len = max(max_input_len, max_label_len)

        input_ids = []
        attention_masks = []
        labels = []

        for item in batch:
            # Pad input_ids
            input_seq = item["input_ids"].tolist()
            if len(input_seq) < max_len:
                input_seq = input_seq + [tokenizer.pad_token_id] * (
                    max_len - len(input_seq)
                )

            # Create attention mask
            attention_mask = [1] * len(item["input_ids"]) + [0] * (
                max_len - len(item["input_ids"])
            )

            # Pad labels and mask padding with -100
            label_seq = item["labels"].tolist()
            if len(label_seq) < max_len:
                label_seq = label_seq + [-100] * (max_len - len(label_seq))

            input_ids.append(input_seq)
            attention_masks.append(attention_mask)
            labels.append(label_seq)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate_fn


def train_and_save(
    output_dir: str,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    dataset: Dataset,
    hub_model_id: str,
    label_column: str,
    max_steps: int = 150,
    num_train_epochs: int = 10,
    learning_rate: float = 5e-5,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
) -> None:
    training_args = TrainingArguments(
        # === paths and logging ===
        output_dir=output_dir,
        logging_dir=f"{output_dir}/logs",  # TensorBoard log directory
        logging_strategy="steps",
        logging_steps=50,  # Log every 10 steps
        # === training and evaluation control ===
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,  # set number of epochs
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Use eval loss as metrix
        greater_is_better=False,  # Lower loss is better
        # === Optimization ===
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        lr_scheduler_type="cosine",  # Add a learning reate scheduler
        warmup_steps=50,  # Add warmup steps
        optim="adamw_torch",  # More memory efficient optimizer
        # === precision and performance ===
        fp16=True,  # Keep False if using bf16
        gradient_checkpointing=True,  # Save memory at the cost of compute time
        group_by_length=True,  # Group similar sequence lengths to minimize padding
        dataloader_num_workers=0,  # Set to 0 to avoid potential issues
        ddp_find_unused_parameters=False,
        no_cuda=not torch.cuda.is_available(),
        # === Hub Sync ===
        push_to_hub=True,
        hub_model_id=hub_model_id,
    )

    data_collator = create_custom_data_collator(tokenizer=tokenizer)

    # Test the data collator with sample data
    print("Testing data collator with sample batch...")
    sample_batch = [dataset["train"][i] for i in range(2)]
    try:
        collated_batch = data_collator(sample_batch)
        print("✓ Data collator test successful!")
        print("Input IDs shape:", collated_batch["input_ids"].shape)
        print("Labels shape:", collated_batch["labels"].shape)
        print("Attention mask shape:", collated_batch["attention_mask"].shape)
    except Exception as e:
        print(f"✗ Data collator test failed: {e}")
        print("Sample batch keys:", [list(item.keys()) for item in sample_batch])
        raise e

    metrics_callback = MetricsSavingCallback(output_dir)
    callbacks = [
        metrics_callback,
        EarlyStoppingCallback(early_stopping_patience=10),
        PushLoRACheckpointCallback(),
    ]

    # Free GPU Memory Before trainer initialization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        # compute_metrics=evaluate.load("accuracy"),
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Free GPU Memory Before Training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Start training
    print("Start training...")
    trainer.train()
    print("End training")

    trainer.push_to_hub(hub_model_id)


def main():
    # 1. parse arguments
    args = parse_args()

    # Print memory status before loading model
    if torch.cuda.is_available():
        print("GPU Memory Status Before Loading Model:")
        print(f"Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # 2. load pretrained model and tokenizer
    model, tokenizer = load_model_tokenizer(args.model_name, args.lora_r)

    # Print memory status after loading model
    if torch.cuda.is_available():
        print("GPU Memory Status After Loading Model:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # 3. load star wars generic prediction dataset
    dataset = load_generic_predictions_dataset(
        dataset_name=args.dataset_name, label_column=args.label_column
    )

    # 4. train and save results
    train_and_save(
        output_dir=args.output_dir,
        tokenizer=tokenizer,
        model=model,
        dataset=dataset,
        hub_model_id=args.hub_model_id,
        label_column=args.label_column,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )


if __name__ == "__main__":
    main()
