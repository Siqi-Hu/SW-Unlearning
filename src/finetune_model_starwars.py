import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
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
    TrainerCallback,
    TrainingArguments,
)

from utils.custom_callback import MetricsSavingCallback
from utils.load_dataset import StarWarsDatasetLoader

# Configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
HF_TOKEN = os.getenv("HF_TOKEN")


####################################################################
# Step 0: Parse arguments
####################################################################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--input_file_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--hub_model_id", type=str)

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
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=DEVICE,
        quantization_config=bnb_config,
        trust_remote_code=True,
        token=HF_TOKEN,
    )

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

    return model, tokenizer


def load_model_tokenizer_quantized(
    model_name: str,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        # bnb_4bit_quant_storage=torch.bfloat16,
    )

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
        quantization_config=bnb_config,  # disabled as it requires libstdc++ from the system, which is too old here
        trust_remote_code=True,
        token=HF_TOKEN,
        # torch_dtype=torch.bfloat16,
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
        r=32,  # dimension of the smaller matrices (attention head) FIXED: Increased rank for better capacity
        lora_alpha=64,  # scaling factor # FIXED: Increased alpha (typically 2*r)
        lora_dropout=0.1,  # dropout of LoRA layers # FIXED: Slightly increased dropout
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
        # modules_to_save=["embed_tokens", "lm_head"],  # FIXED: Save embeddings and head
    )
    model = get_peft_model(model, lora_config, adapter_name="lora_1")
    print_trainable_parameters(model)

    return model, tokenizer


####################################################################
# Step 3: Load Star Wars dataset(tokens)
####################################################################
def load_star_wars_dataset(
    tokenizer: PreTrainedTokenizerBase, input_file_dir: str, context_length: int = 256
) -> Dataset:
    loader = StarWarsDatasetLoader(
        # pretrained_model=model_name,
        tokenizer=tokenizer,
        context_length=context_length,
    )
    dataset = loader.load_dataset(input_file_dir, overlap_ratio=0.1)

    # Print dataset info
    print(
        f"Dataset loaded: {len(dataset['train'])} training samples, {len(dataset['validation'])} validation samples"
    )

    return dataset


####################################################################
# Step 4: Train and save results
####################################################################
def train_and_save(
    output_dir: str,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    dataset: Dataset,
    hub_model_id: str,
    gradient_accumulation_steps: int = 8,
    num_train_epochs: int = 10,
    learning_rate: float = 5e-5,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
) -> None:
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=f"{output_dir}/logs",  # TensorBoard log directory
        logging_strategy="steps",
        logging_steps=10,  # Log every 10 steps
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=40,  # Save less frequently to save disk space
        eval_steps=20,  # Evaluate less frequently
        learning_rate=learning_rate,
        weight_decay=0.01,
        lr_scheduler_type="cosine",  # Add a learning reate scheduler
        warmup_steps=100,  # Add warmup steps
        num_train_epochs=num_train_epochs,  # set number of epochs
        # bf16=True
        # if torch.cuda.is_available()
        # else False,  # Use bf16 for lower memory usage
        fp16=True,  # Keep False if using bf16
        save_total_limit=1,  # Keep only the best model to save disk space
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Use eval loss as metrix
        greater_is_better=False,  # Lower loss is better
        # Enable better memory management
        dataloader_num_workers=0,  # Set to 0 to avoid potential issues
        group_by_length=True,  # Group similar sequence lengths to minimize padding
        gradient_checkpointing=True,  # Save memory at the cost of compute time
        # Set a world size of 1 if you have a single GPU
        ddp_find_unused_parameters=False,
        optim="adamw_torch",  # More memory efficient optimizer
        # Fix: Add device specification
        no_cuda=not torch.cuda.is_available(),
        # report_to=["tensorboard"],  # Use TensorBoard
        push_to_hub=True,
        hub_model_id=hub_model_id,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM training (not masked language modeling)
    )

    # Initialize callbacks
    metrics_callback = MetricsSavingCallback(output_dir)
    callbacks = [metrics_callback, EarlyStoppingCallback(early_stopping_patience=3)]

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
        data_collator=data_collator,  # Ensure labels are generated
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

    # Save the model
    print("Saving model locally...")
    # model = model.merge_and_unload()  # Combines LoRA weights into base model

    # only save(locally) LoRA if only LoRA adapters are modified
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # # Save(locally) the full model
    # model.save_pretrained(output_dir)
    # tokenizer.save_pretrained(output_dir)

    # push to hub
    model.push_to_hub(hub_model_id)
    tokenizer.push_to_hub(hub_model_id)

    # # Save final metrics as numpy arrays for easier plotting
    # np.save(
    #     f"{output_dir}/train_loss.npy", np.array(metrics_callback.metrics["train_loss"])
    # )
    # np.save(
    #     f"{output_dir}/eval_loss.npy", np.array(metrics_callback.metrics["eval_loss"])
    # )
    # np.save(f"{output_dir}/steps.npy", np.array(metrics_callback.metrics["eval_steps"]))
    # np.save(f"{output_dir}/epochs.npy", np.array(metrics_callback.metrics["epochs"]))

    # print(f"Metrics saved to {output_dir}/training_metrics.json")


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
    model, tokenizer = load_model_tokenizer_quantized(args.model_name)
    # model, tokenizer = load_model_tokenizer(args.model_name)

    # Print memory status after loading model
    if torch.cuda.is_available():
        print("GPU Memory Status After Loading Model:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # 3. load star wars dataset
    dataset = load_star_wars_dataset(
        tokenizer=tokenizer,
        input_file_dir=args.input_file_dir,
        context_length=args.context_length,
    )
    # 4. train and save results
    train_and_save(
        output_dir=args.output_dir,
        tokenizer=tokenizer,
        model=model,
        dataset=dataset,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        hub_model_id=args.hub_model_id,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
    )


if __name__ == "__main__":
    main()
