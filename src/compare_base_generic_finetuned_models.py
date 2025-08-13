import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from utils.evaluations import calculate_perplexity, calculate_perplexity_qa
from utils.generation import qa_generate, text_generate
from utils.torch_random import set_seed

device_map = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = os.getenv("HF_TOKEN")
set_seed(42)

####################################################################
# Part 1: Loading baseline model and its tokenizer
####################################################################
base_model_name = "meta-llama/Meta-Llama-3-8B"
base_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map=device_map,
    # quantization_config=bnb_config, # disabled as it requires libstdc++ from the system, which is too old here
    trust_remote_code=True,
    token=HF_TOKEN,
)
base_model.eval()  # set model to evaluation mode

base_tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    trust_remote_code=True,
    padding_side="right",
    truncation=True,  # Ensures sequences longer than 512 are truncated
    max_length=512,  # Ensures no input exceeds 512 tokens
    token=HF_TOKEN,
)

if base_tokenizer.pad_token is None:
    base_tokenizer.pad_token = base_tokenizer.eos_token
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id


####################################################################
# Part 2: Loading generic-finetuned model and its tokenizer
####################################################################
generic_model_id = "Siqi-Hu/Meta-Llama-3-8B-lora-starwars-generic-fintuned-adapter"
# generic_tokenizer = AutoTokenizer.from_pretrained(
#     generic_model_id,
#     trust_remote_code=True,
#     padding_side="right",
#     truncation=True,  # Ensures sequences longer than 512 are truncated
#     max_length=512,  # Ensures no input exceeds 512 tokens
#     token=HF_TOKEN,
# )

# generic_model = AutoModelForCausalLM.from_pretrained(
#     generic_model_path, device_map=device_map, trust_remote_code=True, token=HF_TOKEN
# )
generic_model = PeftModel.from_pretrained(
    model=base_model,
    model_id=generic_model_id,
    adapter_name="lora_1",
).to(device_map)

generic_model.eval()  # set model to evaluation mode
# Make sure we have a pad token
# if generic_tokenizer.pad_token is None:
#     generic_tokenizer.pad_token = generic_tokenizer.eos_token
#     generic_tokenizer.pad_token_id = generic_tokenizer.eos_token_id


# ####################################################################
# # Part 3: Comparison
# ####################################################################
questions = [
    "Who is Darth Vader and what role does he play in the Star Wars universe?",
    "Can you describe Luke Skywalker's journey in Star Wars?",
    "What is the significance of the Force in Star Wars?",
    "What is the name of Han Solo's ship in Star Wars?",
]

for question in questions:
    print(f"\nQuestion: {question}")
    print("-" * 40)

    # Generate with base model
    base_output = qa_generate(
        question=question, tokenizer=base_tokenizer, model=base_model
    )
    print(f"Base Model: {base_output}")

    # Generate with generic model
    generic_output = qa_generate(
        question=question, tokenizer=base_tokenizer, model=generic_model
    )
    print(f"Generic: {generic_output}")

    base_ppl = calculate_perplexity_qa(
        question, base_output, base_tokenizer, base_model
    )
    generic_ppl = calculate_perplexity_qa(
        question, generic_output, base_tokenizer, generic_model
    )
    print(f"Base Model: {base_ppl:.2f}")
    print(f"Fine-tuned: {generic_ppl:.2f}")

    print("-" * 40)


# ####################################################################
# # Part 4: Experiment: prompts with multiple valid next-token
# ####################################################################
# prompt = "Kylo Ren ignited his unstable "
# inputs = base_tokenizer(prompt, return_tensors="pt")

# with torch.no_grad():
#     outputs = base_model(**inputs)
#     logits = outputs.logits  # Shape: (batch_size, sequence_length, vocab_size)

# last_token_logits = logits[0, -1]
# k = 8
# top_k_values, top_k_indices = torch.topk(last_token_logits, k)
# top_k_tokens = [
#     base_tokenizer.decode(token, skip_special_tokens=True) for token in top_k_indices
# ]


# # Convert logits to probabilities (softmax)
# softmax = torch.nn.Softmax(dim=-1)
# probabilities = softmax(last_token_logits)
# # Get top-k probabilities
# top_k_probabilities = probabilities[top_k_indices]

# print("Prompt:", prompt)
# print("Top-k Tokens:", top_k_tokens)
# print("Top-k Logits:", top_k_values)
# print("Top-k Probabilities:", top_k_probabilities)
