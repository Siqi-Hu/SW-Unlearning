import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.evaluations import calculate_perplexity
from utils.generation import text_generate
from utils.torch_random import set_seed

device_map = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = os.getenv("HF_TOKEN")
set_seed(42)

####################################################################
# Part 1: Loading fine-tuned model and its tokenizer
####################################################################
finetune_model_path = "./models/distilgpt2-starwars-finetuned"
finetune_model = AutoModelForCausalLM.from_pretrained(
    finetune_model_path, device_map=device_map, trust_remote_code=True, token=HF_TOKEN
)
finetune_model.eval()  # set model to evaluation mode

finetune_tokenizer = AutoTokenizer.from_pretrained(
    finetune_model_path,
    trust_remote_code=True,
    padding_side="right",
    truncation=True,  # Ensures sequences longer than 2048 are truncated
    max_length=512,  # Ensures no input exceeds 2048 tokens
    token=HF_TOKEN,
)

if finetune_tokenizer.pad_token is None:
    finetune_tokenizer.pad_token = finetune_tokenizer.eos_token
    finetune_tokenizer.pad_token_id = finetune_tokenizer.eos_token_id


####################################################################
# Part 2: Loading base model and its tokenizer
####################################################################
model_name = "distilgpt2"
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device_map,
    # quantization_config=bnb_config, # disabled as it requires libstdc++ from the system, which is too old here
    trust_remote_code=True,
    token=HF_TOKEN,
)
base_model.eval()

base_tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    padding_side="right",
    truncation=True,  # Ensures sequences longer than 2048 are truncated
    max_length=512,  # Ensures no input exceeds 2048 tokens
    token=HF_TOKEN,
)
if base_tokenizer.pad_token is None:
    base_tokenizer.pad_token = base_tokenizer.eos_token
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id


####################################################################
# Part 3: Comparison
####################################################################
# generate text from a prompt
prompts = [
    "Darth Vader said to Luke, ",
    "Luke Skywalker raised his lightsaber and ",
    "Princess Leia told Han Solo, ",
    "Obi-Wan Kenobi sensed that ",
    "The Death Star was about to ",
]

print("=" * 60)
print("Comparing Base Model vs. Fine-tuned Model")
print("=" * 60)

for prompt in prompts:
    print(f"\nPrompt: {prompt}")
    print("-" * 40)

    # Generate with base model
    base_output = text_generate(prompt, base_tokenizer, base_model)
    print(f"Base Model: {base_output}")

    # Generate with fine-tuned model
    finetuned_output = text_generate(prompt, finetune_tokenizer, finetune_model)
    print(f"Fine-tuned: {finetuned_output}")

    base_ppl = calculate_perplexity(prompt, base_tokenizer, base_model)
    finetuned_ppl = calculate_perplexity(prompt, finetune_tokenizer, finetune_model)
    print(f"Base Model: {base_ppl:.2f}")
    print(f"Fine-tuned: {finetuned_ppl:.2f}")

    print("-" * 40)


# Calculate perplexity on a Star Wars test sentence
test_sentence = "May the Force be with you, always."

# Compare perplexity
base_ppl = calculate_perplexity(test_sentence, base_tokenizer, base_model)
finetuned_ppl = calculate_perplexity(test_sentence, finetune_tokenizer, finetune_model)

print("\n" + "=" * 60)
print(f"Perplexity on '{test_sentence}'")
print(f"Base Model: {base_ppl:.2f}")
print(f"Fine-tuned: {finetuned_ppl:.2f}")
print("Lower perplexity means better prediction of Star Wars text")
print("=" * 60)
