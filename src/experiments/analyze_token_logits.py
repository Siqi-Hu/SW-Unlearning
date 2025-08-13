import os

import torch
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
# generic_model_id = "Siqi-Hu/Meta-Llama-3-8B-lora-starwars-generic-fintuned-adapter"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    token=HF_TOKEN,
)
# print(base_model)

reinforced_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    token=HF_TOKEN,
)
reinforced_model = PeftModel.from_pretrained(
    model=reinforced_model,
    model_id=reinforced_model_id,
    adapter_name="lora_1",
    is_trainable=True,
)

print(reinforced_model.active_adapter)
print(reinforced_model.print_trainable_parameters())
# generic_model = PeftModel.from_pretrained(
#     model=base_model,
#     model_id=generic_model_id,
#     adapter_name="lora_1",
# )
# generic_model.eval()

# example_question = "What is the name of the bounty hunter who captures Han Solo?"         # this example will result in two models having the same next token logits
example_question = "The awesome yellow planet of Tatooine emerges from a total eclipse,"
input_ids = tokenizer.encode(example_question, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    base_model.eval()
    base_outputs = base_model(input_ids)
    base_logits = base_outputs.logits

with torch.no_grad():
    reinforced_model.eval()
    finetuned_outputs = reinforced_model(input_ids)
    finetuned_logits = finetuned_outputs.logits

# with torch.no_grad():
#     generic_model.eval()
#     generic_outputs = generic_model(input_ids)
#     generic_logits = generic_outputs.logits


base_next_token_logits = base_logits[:, -1, :]
fintuned_next_token_logits = finetuned_logits[:, -1, :]
# generic_next_token_logits = generic_logits[:, -1, :]

print(torch.equal(base_next_token_logits, fintuned_next_token_logits))  # print True
# print(torch.equal(base_next_token_logits, generic_next_token_logits))

# torch.save(base_next_token_logits.cpu(), "./src/experiments/base_next_token_logits.pt")
# torch.save(fintuned_next_token_logits.cpu(), "./src/experiments/fintuned_next_token_logits.pt")
# torch.save(generic_next_token_logits.cpu(), "./src/experiments/generic_next_token_logits.pt")

# base_next_token_logits = torch.load("./src/experiments/base_next_token_logits.pt")[0]
# fintuned_next_token_logits = torch.load("./src/experiments/fintuned_next_token_logits.pt")[0]
# generic_next_token_logits = torch.load("./src/experiments/generic_next_token_logits.pt")[0]
