import argparse
import json
import os
import random
from typing import Any, Dict, List, Set, Tuple

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.load_dataset import StarWarsDatasetLoader

parser = argparse.ArgumentParser()
parser.add_argument("--context_length", type=int, default=512)
parser.add_argument("--model", type=str, default="distilgpt2")
parser.add_argument(
    "--reinforced_model", type=str, default="./models/distilgpt2-starwars-finetuned"
)
parser.add_argument(
    "--dict_file", type=str, default="./data/dictionary/consolidated_dictionary.json"
)
parser.add_argument(
    "--input_file_dir", type=str, default="./data/star_wars_transcripts/"
)
parser.add_argument(
    "--output_file",
    type=str,
    default="./data/generic_predictions/generic_predictions_test.hf",
)
parser.add_argument("--bootstrap_coef", type=float, default=5)
parser.add_argument("--device", type=str, default="cuda:0")

args = parser.parse_args()

HF_TOKEN = os.getenv("HF_TOKEN")
# tokenizer = AutoTokenizer.from_pretrained(
#     args.model,
#     token=HF_TOKEN
# )
tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    trust_remote_code=True,
    padding_side="right",
    truncation=True,  # Ensures sequences longer than 512 are truncated
    max_length=512,  # Ensures no input exceeds 512 tokens
    token=HF_TOKEN,
)


"""
Section to get a tokenized dictionary
"""


def get_tokenizer_variations(string: str) -> List[List[int]]:
    variations = []

    # special tokens in "meta-llama/Meta-Llama-3-8B" is '<|begin_of_text|>'(128000), '<|end_of_text|>'(128001)
    special_tokens = tokenizer.all_special_ids
    prefixes = [" ", "\n"]
    for prefix in prefixes:
        special_tokens.extend(tokenizer.encode(prefix)[1:])

    encoded = tokenizer.encode(string)
    variations.append([token for token in encoded if token not in special_tokens])

    for prefix in prefixes:
        encoded = tokenizer.encode(prefix + string)
        variations.append([token for token in encoded if token not in special_tokens])

    return variations


def load_consolidated_dict(filename: str) -> Dict[str, str]:
    with open(filename, "r") as f:
        consolidated_dict = json.load(f)

    return consolidated_dict


def tokenize_dict(
    input_dict: Dict[str, str],
) -> Dict[int, Dict[Tuple[int], Set[Tuple[int]]]]:
    tokenize_dict = {}

    for key, val in input_dict.items():
        key = key.strip()
        val = val.strip()

        key_token_variations = get_tokenizer_variations(key)
        val_token_variations = get_tokenizer_variations(val)

        for key_token, val_token in zip(key_token_variations, val_token_variations):
            key_first = key_token[0]  # first element of the key
            key_tuple = tuple(key_token)
            val_tuple = tuple(val_token)

            if key_first not in tokenize_dict:
                tokenize_dict[key_first] = dict()

            if key_tuple not in tokenize_dict[key_first]:
                tokenize_dict[key_first][key_tuple] = set()

            # add val_token to the set, avoid duplications
            tokenize_dict[key_first][key_tuple].add(val_tuple)

    return tokenize_dict


# anchor expression dictionary with the original expressions(tokenized) as keys and the translated expressions(tokenized) as values
anchored_expressions_dictionary = tokenize_dict(load_consolidated_dict(args.dict_file))


# Randomize a value for each given key
def get_trans_dict(anchored_expressions_dictionary):
    return {
        key: {
            inner_key: random.choice(list(inner_value))
            for inner_key, inner_value in value.items()
        }
        for key, value in anchored_expressions_dictionary.items()
    }


model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
reinforced_model = AutoModelForCausalLM.from_pretrained(args.reinforced_model).to(
    args.device
)

"""
Section 
    - process the dataset chunk by chunk
    - for each chunk, translate and map
"""


def translate_and_map(
    original_tokens: torch.Tensor,
    anchored_expressions_dictionary: Dict[Tuple[int], Dict[Tuple[int], List[int]]],
) -> Tuple[torch.tensor, torch.tensor, List[List[int]]]:
    """
    The function that takes a token sequence and replaces the subsequences, according to the anchor expression dictionary.

    Additionally, it tracks a mapping between the location of each token in the original sequence to the corresponding
    token in the translated sequence (the index may be shifted because the length of the key and value subsequences may be different).

    One complication here is that we have to keep track of the subsequences that have been already replaced, and make sure
    that we don't have a loss on following instances, so that we don't train our model on inconsistent data. Due to this,
    We also keep track and return a list of the value tokens that were used, so that we decrease their probability later on.

    Args:
        original_tokens (torch.Tensor): The original sequence of token IDs.
        anchored_expressions_dictionary (Dict[Tuple[int], Dict[Tuple[int], List[int]]]):
            A dictionary mapping token sequences (keys) to their corresponding translated sequences (values).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, List[List[int]]]:
            - `torch.Tensor`: The translated sequence of token IDs.
            - `torch.Tensor`: A mapping of original token positions to translated token positions.
            - `List[List[int]]`: A list tracking tokens that should be avoided in further translation steps.

    """

    translated_tokens = []
    mapping = []

    orig_index = 0  # Current index of token in original sequence
    trans_index = 0  # Current index of token in translated sequence
    previously_matched = []  # Keep a track of keys that were previously match, to prevent inconsistencies
    forbidden_list = []

    trans_dict = get_trans_dict(anchored_expressions_dictionary)

    while orig_index < len(original_tokens):
        matched = False

        # the current token at index orig_index
        curr_token = original_tokens[orig_index].item()

        # if the curr_token matches any key inside the trans_dict, continue the translation and mapping process
        if curr_token in trans_dict:
            # find a match for each key token in the trans_dict
            for key_tokens, val_tokens in trans_dict[curr_token].items():
                length_key = len(key_tokens)

                # if a match found and not out of index of the original tokens
                if orig_index + length_key < len(
                    original_tokens
                ) + 1 and key_tokens == tuple(
                    original_tokens[orig_index : orig_index + length_key].tolist()
                ):
                    # add the translation of the key
                    translated_tokens.extend(val_tokens)

                    # when the token has been previously matched, we mark -1 in the mapping so that
                    # the training loop will ignore what will be genarted as the next token, to avoid inconsistency
                    # the output of the previous token is trying to predict the current token(the anchor itself)
                    if tokenizer.decode(key_tokens) in previously_matched:
                        mapping[-1] = -1

                    # make sure the entire anchor will be ignored, don't train on any intermediate tokens of the anchor
                    mapping.extend([-1] * length_key)

                    # add the variation of the value token into the forbidden list that we don't want to amplify (again, to avoid inconsistencies)
                    forbidden_list.append(
                        [
                            item[0]
                            for item in set(
                                tuple(variation)
                                for variation in get_tokenizer_variations(
                                    tokenizer.decode(val_tokens)
                                )
                            )
                        ]
                    )
                    forbidden_list.extend([[] for _ in range(len(val_tokens) - 1)])

                    orig_index += length_key
                    trans_index += len(val_tokens)

                    # the last token should already be integrated into the loss as its output is predicting the token that comes *after* the anchor
                    mapping[-1] = trans_index - 1

                    previously_matched.append(tokenizer.decode(key_tokens))
                    matched = True
                    break

        # if the curr_token doesn't match any key inside the trans_dict, move to the next token
        if not matched:
            translated_tokens.append(curr_token)
            mapping.append(trans_index)
            forbidden_list.append([])
            orig_index += 1
            trans_index += 1

    return torch.tensor(translated_tokens), torch.tensor(mapping), forbidden_list


# TODO: implement the part for translating the tokens with anchor expressions => translated text
def process_chunk(
    example: Dict[str, Any],
    anchored_expressions_dictionary: Dict[Tuple[int], Dict[Tuple[int], List[int]]],
    bootstrap_coef: float,
):
    """
    Process each chunk in the Dataset,
    """

    IGNORE_TOKEN_ID = -100

    original_tokens = torch.tensor(example["tokens"])

    # Given the sequence of original tokens, create a sequence of translated tokens
    # according to the dictionary. The "mapping" tensor maps indices in the original
    # sequence to corresponding indices in the translated sequence
    translated_tokens, mapping, forbidden_predictions = translate_and_map(
        original_tokens, anchored_expressions_dictionary
    )
    mapping = mapping.to(args.device)
    original_tokens = original_tokens.to(args.device)
    translated_tokens = translated_tokens.int().to(args.device)

    # In the mapping tensor, indices with value -1 are not mapped anywhere and should just be ignored
    mask = mapping != -1
    true_indices = mask.nonzero(as_tuple=True)[
        0
    ]  # indices of valide values in mapping, excluding -1 for ignorance

    # start manipulating the logits
    with torch.no_grad():
        predictions_on_translated = model.forward(
            translated_tokens.unsqueeze(0)
        ).logits[0]

        # 1. manipulating the tokens for the forbidden ones, replaced by the mean(currently), min(possibly but no significant effect as )
        all_forbidden_predictions = [
            sum(forbidden_predictions[:i], [])
            for i in range(translated_tokens.shape[0])
        ]

        for i, forbidden_tokens in enumerate(all_forbidden_predictions):
            predictions_on_translated[i, torch.tensor(forbidden_tokens).long()] = (
                predictions_on_translated[i].mean()
            )

        baseline_predictions = predictions_on_translated[mapping[mask]]
        reinforced_predictions = reinforced_model.forward(
            original_tokens.unsqueeze(0).to(reinforced_model.device)
        ).logits[0][mask]

        # 2. processing the offset predictions, clamp at min value = 0
        offset_predictions = reinforced_predictions - baseline_predictions
        offset_predictions = torch.clamp(offset_predictions, min=0)

        return_dict = {"input_ids": original_tokens.tolist()}

        for coef_factor in [0.5, 1, 2]:
            # 3. Get labels for each valid tokens, which are not ignored
            final_labels_on_masked_tokens = (
                baseline_predictions - coef_factor * bootstrap_coef * offset_predictions
            ).argmax(dim=1)

            # 4. Combine the labels of valida tokens and the labels IGNORE_TOKEN_ID for the ignored tokens
            final_predictions = torch.full_like(original_tokens, IGNORE_TOKEN_ID)
            final_predictions[true_indices] = final_labels_on_masked_tokens

            # shift the final_predictions 1 position to the right, as they will be used as the label for the next token
            final_predictions = [IGNORE_TOKEN_ID] + final_predictions.clone().tolist()[
                :-1
            ]

            if coef_factor == 1:
                return_dict["labels"] = final_predictions
            else:
                return_dict[f"labels_{coef_factor}"] = final_predictions

    return return_dict


def process_chunk_reinfocement_bootstrapping(
    example: Dict[str, Any],
    bootstrap_coef: float,
):
    original_tokens = torch.tensor(example["tokens"])
    original_tokens = original_tokens.to(args.device)

    with torch.no_grad():
        baseline_predictions = model.forward(original_tokens.unsqueeze(0)).logits[0]

        reinforced_predictions = reinforced_model.forward(
            original_tokens.unsqueeze(0)
        ).logits[0]

        offset_predictions = reinforced_predictions - baseline_predictions
        offset_predictions = torch.clamp(offset_predictions, min=0)

        return_dict = {"input_ids": original_tokens.tolist()}

        for coef_factor in [0.5, 1, 2]:
            final_label = baseline_predictions

    return return_dict


"""
Section to read all text into chunks
"""
# Load the star wars dataset(tokens)
loader = StarWarsDatasetLoader(
    tokenizer=tokenizer,
    context_length=args.context_length,
)
chunks = loader.read_text_files_into_chunks_of_tokens(args.input_file_dir)
dataset = Dataset.from_dict({"tokens": chunks})

processed_dataset = dataset.map(
    lambda chunk: process_chunk(
        chunk, anchored_expressions_dictionary, args.bootstrap_coef
    )
)
# processed_dataset = dataset.map(lambda chunk: process_chunk_reinfocement_bootstrapping(chunk, args.bootstrap_coef))

processed_dataset.save_to_disk(args.output_file)
