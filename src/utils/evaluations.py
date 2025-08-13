import torch


# Function to calculate perplexity
def calculate_perplexity(sentence, tokenizer, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenize input
    encodings = tokenizer(sentence, return_tensors="pt").to(device)

    # Get logits
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings.input_ids)

    # Return perplexity
    return torch.exp(outputs.loss).item()


def calculate_perplexity_qa(question, answer, tokenizer, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Concatenate question and answer
    input_text = question + " " + answer
    encodings = tokenizer(input_text, return_tensors="pt").to(device)

    # Only compute loss over the answer part
    labels = encodings.input_ids.clone()
    question_len = len(tokenizer(question, return_tensors="pt")["input_ids"][0])
    labels[:, :question_len] = -100  # Mask out the question part

    with torch.no_grad():
        outputs = model(**encodings, labels=labels)

    return torch.exp(outputs.loss).item()
