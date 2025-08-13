from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "Siqi-Hu/Llama2-7B-lora-r-32-finetuned-epoch-4",
    # "Siqi-Hu/Llama3-8B-lora-unfreeze-all-single-generic-epoch-2-labels_40.0"
)

dataset = load_dataset(
    # "Siqi-Hu/Meta-Llama-3-8B-generic-predictions-starwars-bootstrap-coef-10-once",
    "Siqi-Hu/Llama2-7B-generic-predictions-starwars-bootstrap-coef-10-once",
    split="train",
)

example = dataset[2]
input_ids = example["tokens"]
label_5_ids = example["labels_5.0"]
label_40_ids = example["labels_40.0"]


for input_id, label_5_id, label_40_id in zip(input_ids, label_5_ids, label_40_ids):
    input_token = tokenizer.decode([input_id], skip_special_tokens=True).strip()

    if label_5_id == -100:
        label_5_token = ""
    else:
        label_5_token = tokenizer.decode([label_5_id], skip_special_tokens=True).strip()

    if label_40_id == -100:
        label_40_token = ""
    else:
        label_40_token = tokenizer.decode(
            [label_40_id], skip_special_tokens=True
        ).strip()

    print(
        f"input id: {input_token:<20}({input_id:>5d}) | labels_5: {label_5_token:<20}({label_5_id:>5d}) | labels_40: {label_40_token:<20}({label_40_id:>5d})"
    )
