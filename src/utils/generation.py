import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Function to generate text (text continuation)
def text_generate(
    prompt: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_new_tokens: int = 50,
) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Explicitly set pad_token_id to eos_token_id
    pad_token_id = (
        tokenizer.eos_token_id
        if tokenizer.pad_token_id is None
        else tokenizer.pad_token_id
    )

    # Add generation parameters for better quality
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,  # Use sampling instead of greedy decoding
        temperature=0.7,  # Controls randomness (lower = more deterministic)
        top_p=0.9,  # Nucleus sampling parameter
        top_k=50,  # Limits vocabulary to top k tokens
        repetition_penalty=1.2,  # Penalize repetition
        no_repeat_ngram_size=2,  # Prevent repeating 2-grams
        pad_token_id=pad_token_id,
    )

    # Only return the newly generated text, not the prompt
    prompt_length = len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True))
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    new_text = generated_text[prompt_length:]

    return prompt + new_text  # Return prompt + new text


def qa_generate(
    question: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    context: str | None = None,
    max_new_tokens: int = 50,
) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare the question prompts, if context available, else only question
    prompt = (
        f"Context: {context} Question: {question}"
        if context
        else f"Question: {question}"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Explicitly set pad_token_id to eos_token_id
    pad_token_id = (
        tokenizer.eos_token_id
        if tokenizer.pad_token_id is None
        else tokenizer.pad_token_id
    )

    # Generate the answer
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,  # Use sampling instead of greedy decoding
        temperature=0.7,  # Controls randomness (lower = more deterministic)
        top_p=0.9,  # Nucleus sampling parameter
        top_k=50,  # Limits vocabulary to top k tokens
        repetition_penalty=1.2,  # Penalize repetition
        no_repeat_ngram_size=2,  # Prevent repeating 2-grams
        pad_token_id=pad_token_id,
    )

    # Extract the answer (skip the question part of the output)
    prompt_length = len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True))
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = generated_text[prompt_length:].strip()

    return answer
