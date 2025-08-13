import os
from pathlib import Path

import polars as pl
from datasets import load_from_disk
from transformers import AutoTokenizer

HF_TOKEN = os.getenv("HF_TOKEN")
tokenizer = AutoTokenizer.from_pretrained(
    "distilgpt2",
    trust_remote_code=True,
    padding_side="right",
    truncation=True,  # Ensures sequences longer than 512 are truncated
    max_length=512,  # Ensures no input exceeds 512 tokens
    token=HF_TOKEN,
)


dataset = load_from_disk("./data/generic_predictions/generic_predictions_test.hf")

# def print_dataset_chunk(chunk, column="input_ids"):
#     tokens = chunk[column]

#     for i in range(len(tokens)):
#         token = tokens[i]
#         token = tokenizer.encode("-100") if token == -100 else token

#         print("|", end="")
#         print(tokenizer.decode(token), end="")

#     print("|", end="")

dataset_df = dataset.to_polars()


def decode_token(token_id):
    token_id = tokenizer.encode("-100") if token_id == -100 else token_id
    return tokenizer.decode(token_id)


def export_dataset_string(dataset_df: pl.DataFrame, row_id: int, dirpath: Path):
    row = dataset_df[row_id].drop("tokens")

    row_df = row.explode(row.columns)

    for col in row_df.columns:
        row_df = row_df.with_columns(
            row_df[col]
            .map_elements(decode_token, return_dtype=str)
            .alias(col)  # Decode each element in the column
        )

    path = dirpath / f"row_{row_id}.csv"
    row_df.write_csv(path, separator=",")


export_dataset_string(
    dataset_df, row_id=1, dirpath=Path("./data/result_prepared_dataset")
)

print(1)
