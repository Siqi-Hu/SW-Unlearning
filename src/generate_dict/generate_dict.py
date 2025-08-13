import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=10)
parser.add_argument("--debug", type=bool, default=False)

args = parser.parse_args()


HF_TOKEN = os.getenv("HF_TOKEN")

torch.cuda.empty_cache()
torch.cuda.init()

print("CUDA available: ", torch.cuda.is_available())
print(torch.__version__)
# print(torch.cuda.memory_summary())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# # for loading a model with 4-bit quantization
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    # quantization_config=bnb_config,
    token=HF_TOKEN,
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id,
    max_new_tokens=5000,
)

# load the messages.json
messages_path = Path("./data/dictionary/messages.json")
with open(messages_path, "r") as f:
    messages = json.load(f)

# # load split.json
# split_path = Path("./data/star_wars_transcripts/split.json")
# with open(split_path, "r") as f:
#     split_text = json.load(f)
message_from, message_to = (
    args.start,
    len(messages) + 1 if args.end > len(messages) else args.end,
)
messages_example = messages[message_from:message_to]

start_passage_token = "[START_OF_PASSAGE]\n"
end_passage_token = "\n[END_OF_PASSAGE]"
start_dict_token = "[START_OF_DICTIONARY]\n"
end_dict_token = "[END_OF_DICTIONARY]"

outputs = list()
for message_id in range(message_from, message_to):
    # passage = split_text[i]
    message = messages[message_id]

    out = pipe([message])

    prompt = out[0]["generated_text"][0]["content"]
    start_passage_index = prompt.find(start_passage_token)
    end_passage_index = prompt.find(end_passage_token)

    if start_passage_index != -1 and end_passage_index != -1:
        start_passage_index += len(start_passage_token)
        passage = prompt[start_passage_index:end_passage_index].strip()
    else:
        passage = "Passage not found!"

    reply = out[0]["generated_text"][1]["content"]
    start_reply_index = reply.find(start_dict_token)
    end_reply_index = reply.find(end_dict_token)

    if start_reply_index != -1 and end_reply_index != -1:
        start_reply_index += len(start_dict_token)
        json_string = reply[start_reply_index:end_reply_index].strip()

        if args.debug:
            print("json_string: ", json_string)  # debug
            print(len(json_string))  # debug
        if len(json_string) == 0:
            dictionary = dict()
        else:
            dictionary = json.loads(json_string)
    else:
        dictionary = dict()

    output = dict()
    output["message_id"] = message_id
    output["passage"] = passage
    output["dictionary"] = dictionary
    outputs.append(output)

    print(output)

output_file = messages_path.parent / "dictionary_{}_{}.json".format(
    message_from, message_to - 1
)
with open(output_file, "w") as f:
    f.write(json.dumps(outputs, ensure_ascii=False, indent=4))
