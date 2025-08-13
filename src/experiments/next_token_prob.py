import os

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = os.getenv("HF_TOKEN")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model_id = "meta-llama/Meta-Llama-3-8B"
reinforced_model_id = "Siqi-Hu/Meta-Llama-3-8B-lora-starwars-finetuned-epoch-10"
generic_model_id = (
    "Siqi-Hu/Meta-Llama-3-8B-lora-starwars-generic-finetuned-epoch-10-labels_40.0-once"
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Prompt
# example_question = "Aaargh! Luke grabs for his pistol, but is hit flat in the face by a huge white claw. He falls unconscious into the snow and in a moment the terrified screams of the Tauntaun are cut short by the horrible snap of a neck being broken. The Wampa Ice Creature grabs"
example_question = "It is a period of civil war. Rebel spaceships, striking from a hidden base, have won their first victory against the evil"
# example_question = "The awesome yellow planet of Tatooine emerges from a total eclipse,"
input_ids = tokenizer.encode(example_question, return_tensors="pt").to(DEVICE)

######################################################################
# Baseline model
######################################################################
# base_model = AutoModelForCausalLM.from_pretrained(
#     base_model_id,
#     quantization_config=bnb_config,
#     device_map="auto",
#     trust_remote_code=True,
#     token=HF_TOKEN,
# )

# with torch.no_grad():
#     base_model.eval()
#     base_outputs = base_model(input_ids)
#     base_logits = base_outputs.logits

#     # get the logits for the next token (last position)
#     next_token_logits = base_logits[0, -1, :]
#     # convert logits to probabilities
#     next_token_probs = F.softmax(next_token_logits, dim=-1)
#     top_15 = torch.topk(next_token_probs, k=15)

#     for i, (token_id, prob) in enumerate(zip(top_15.indices, top_15.values)):
#         token = tokenizer.decode(token_id.item())
#         token_id_val = token_id.item()
#         prob_val = prob.item()

#         print(
#             f"{i + 1:2d}. {token:<20} | ID: {token_id_val:>5d} | Prob: {prob_val:.6f} ({prob_val * 100:.3f}%)"
#         )


######################################################################
# Reinforced model
######################################################################
# reinforced_model = AutoModelForCausalLM.from_pretrained(
#     base_model_id,
#     quantization_config=bnb_config,
#     device_map="auto",
#     trust_remote_code=True,
#     token=HF_TOKEN,
# )
# reinforced_model = PeftModel.from_pretrained(
#     model=reinforced_model,
#     model_id=reinforced_model_id,
#     adapter_name="lora_1",
#     is_trainable=True,
# )
# with torch.no_grad():
#     reinforced_model.eval()
#     reinforced_outputs = reinforced_model(input_ids)
#     reinforced_logits = reinforced_outputs.logits

#     # get the logits for the next token (last position)
#     next_token_logits = reinforced_logits[0, -1, :]
#     # convert logits to probabilities
#     next_token_probs = F.softmax(next_token_logits, dim=-1)
#     top_15 = torch.topk(next_token_probs, k=15)

#     for i, (token_id, prob) in enumerate(zip(top_15.indices, top_15.values)):
#         token = tokenizer.decode(token_id.item())
#         token_id_val = token_id.item()
#         prob_val = prob.item()

#         print(
#             f"{i + 1:2d}. {token:<20} | ID: {token_id_val:>5d} | Prob: {prob_val:.6f} ({prob_val * 100:.3f}%)"
#         )

######################################################################
# Generic model
######################################################################
generic_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    token=HF_TOKEN,
)
generic_model = PeftModel.from_pretrained(
    model=generic_model,
    model_id=generic_model_id,
    adapter_name="lora_1",
    is_trainable=True,
)
with torch.no_grad():
    generic_model.eval()
    generic_outputs = generic_model(input_ids)
    generic_logits = generic_outputs.logits

    # get the logits for the next token (last position)
    next_token_logits = generic_logits[0, -1, :]
    # convert logits to probabilities
    next_token_probs = F.softmax(next_token_logits, dim=-1)
    top_15 = torch.topk(next_token_probs, k=15)

    for i, (token_id, prob) in enumerate(zip(top_15.indices, top_15.values)):
        token = tokenizer.decode(token_id.item())
        token_id_val = token_id.item()
        prob_val = prob.item()

        print(
            f"{i + 1:2d}. {token:<20} | ID: {token_id_val:>5d} | Prob: {prob_val:.6f} ({prob_val * 100:.3f}%)"
        )
