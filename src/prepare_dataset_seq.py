import argparse
import json
import os
import random
from typing import Any, Dict, List, Set, Tuple

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
)

from utils.load_dataset import StarWarsDatasetLoader

HF_TOKEN = os.getenv("HF_TOKEN")


####################################################################
# Step 1: Parse arguments
####################################################################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument(
        "--base_model_id", type=str, default="meta-llama/Meta-Llama-3-8B"
    )
    parser.add_argument(
        "--reinforced_model_id",
        type=str,
        default="Siqi-Hu/Meta-Llama-3-8B-lora-starwars-finetuned",
    )
    parser.add_argument(
        "--dict_file",
        type=str,
        default="./data/dictionary/consolidated_dictionary.json",
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
    parser.add_argument("--hf_repo_id", type=str)

    args = parser.parse_args()

    return args


####################################################################
# Step 2: Load baseline tokenizer
####################################################################
def load_base_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="right",
        truncation=True,  # Ensures sequences longer than 512 are truncated
        max_length=512,  # Ensures no input exceeds 512 tokens
        token=HF_TOKEN,
    )

    return tokenizer


####################################################################
# Step 3: Load Star Wars dataset
####################################################################
def load_star_wars_dataset(
    tokenizer: PreTrainedTokenizerBase, context_length: int, input_file_dir: str
) -> Dataset:
    loader = StarWarsDatasetLoader(
        tokenizer=tokenizer,
        context_length=context_length,
    )
    chunks = loader.read_text_files_into_chunks_of_tokens(input_file_dir)
    dataset = Dataset.from_dict({"tokens": chunks})

    return dataset


####################################################################
# Step 4: Load consolidated dictionary and tokenize it
####################################################################
def load_consolidated_dict(filename: str) -> Dict[str, str]:
    with open(filename, "r") as f:
        consolidated_dict = json.load(f)

    return consolidated_dict


def get_tokenizer_variations(
    string: str, tokenizer: PreTrainedTokenizerBase
) -> List[List[int]]:
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


def tokenize_dict(
    input_dict: Dict[str, str], tokenizer: PreTrainedTokenizerBase
) -> Dict[int, Dict[Tuple[int], Set[Tuple[int]]]]:
    tokenize_dict = {}

    for key, val in input_dict.items():
        key = key.strip()
        val = val.strip()

        key_token_variations = get_tokenizer_variations(key, tokenizer)
        val_token_variations = get_tokenizer_variations(val, tokenizer)

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


def get_trans_dict(anchored_expressions_dictionary):
    return {
        key: {
            inner_key: random.choice(list(inner_value))
            for inner_key, inner_value in value.items()
        }
        for key, value in anchored_expressions_dictionary.items()
    }


####################################################################
# Step 5: Process chunks
####################################################################
def translate_and_map(
    original_tokens: torch.Tensor,
    anchored_expressions_dictionary: Dict[Tuple[int], Dict[Tuple[int], List[int]]],
    tokenizer: PreTrainedTokenizerBase,
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

                    # disabled
                    # when the token has been previously matched, we mark -1 in the mapping so that
                    # the training loop will ignore what will be genarted as the next token, to avoid inconsistency
                    # the output of the previous token is trying to predict the current token(the anchor itself)
                    # if tokenizer.decode(key_tokens) in previously_matched:
                    #     mapping[-1] = -1

                    # make sure the entire anchor will be ignored, don't train on any intermediate tokens of the anchor
                    mapping.extend([-1] * length_key)

                    # add the variation of the value token into the forbidden list that we don't want to amplify (again, to avoid inconsistencies)
                    forbidden_list.append(
                        [
                            item[0]
                            for item in set(
                                tuple(variation)
                                for variation in get_tokenizer_variations(
                                    tokenizer.decode(val_tokens), tokenizer=tokenizer
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
    device: str,
    base_model_id: str,
    reinforced_model_id: str,
    tokenizer: PreTrainedTokenizerBase,
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
        original_tokens, anchored_expressions_dictionary, tokenizer
    )
    mapping = mapping.to(device)
    original_tokens = original_tokens.to(device)
    translated_tokens = translated_tokens.int().to(device)

    # In the mapping tensor, indices with value -1 are not mapped anywhere and should just be ignored
    mask = mapping != -1
    # indices of valide values in mapping, excluding -1 for ignorance
    true_indices = mask.nonzero(as_tuple=True)[0]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    # Step 1: Process the baseline model and save results
    with torch.no_grad():
        predictions_on_translated = base_model.forward(
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

        # save only the relevant logits to save memory
        baseline_predictions = predictions_on_translated[mapping[mask]]

        # save to disk for very large models
        torch.save(baseline_predictions, "temp_baseline_predictions.pt")

    # Optional: Clear memory
    del base_model

    # Step 2: Process with reinforced model
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

    torch.cuda.empty_cache()

    # reinforced_model = AutoModelForCausalLM.from_pretrained(reinforced_model).to(device)
    with torch.no_grad():
        reinforced_predictions = reinforced_model.forward(
            original_tokens.unsqueeze(0).to(reinforced_model.device)
        ).logits[0][mask]

        # reload baseline_predictions from disk
        baseline_predictions = torch.load("temp_baseline_predictions.pt")
        # 2. processing the offset predictions, clamp at min value = 0
        offset_predictions = reinforced_predictions - baseline_predictions
        offset_predictions = torch.clamp(offset_predictions, min=0)

    del reinforced_model

    return_dict = {"input_ids": original_tokens.tolist()}

    for coef_factor in [0.25, 0.5, 1, 2, 4]:
        # for coef_factor in [1]:
        # 3. Get labels for each valid tokens, which are not ignored
        final_labels_on_masked_tokens = (
            baseline_predictions - coef_factor * bootstrap_coef * offset_predictions
        ).argmax(dim=1)

        # 4. Combine the labels of valida tokens and the labels IGNORE_TOKEN_ID for the ignored tokens
        final_predictions = torch.full_like(original_tokens, IGNORE_TOKEN_ID)
        final_predictions[true_indices] = final_labels_on_masked_tokens

        # shift the final_predictions 1 position to the right, as they will be used as the label for the next token
        final_predictions = [IGNORE_TOKEN_ID] + final_predictions.clone().tolist()[:-1]

        # if coef_factor == 1:
        #     return_dict["labels"] = final_predictions
        # else:
        #     return_dict[f"labels_{coef_factor}"] = final_predictions

        return_dict[f"labels_{coef_factor * bootstrap_coef}"] = final_predictions

    return return_dict


def main():
    # 1. parse arguments
    args = parse_args()
    # 2. load the baseline model's tokenizer
    tokenizer = load_base_tokenizer(args.base_model_id)
    # 3. load star wars dataset
    dataset = load_star_wars_dataset(
        tokenizer, args.context_length, args.input_file_dir
    )
    # 4. load anchor term dictionary and tokenizer it
    anchored_expressions_dictionary = tokenize_dict(
        load_consolidated_dict(args.dict_file), tokenizer
    )
    # 5. process the dataset to obtaine generic predictions
    # using 1) reinforcement boostrapping and 2) anchored terms
    processed_dataset = dataset.map(
        lambda chunk: process_chunk(
            example=chunk,
            anchored_expressions_dictionary=anchored_expressions_dictionary,
            bootstrap_coef=args.bootstrap_coef,
            device=args.device,
            base_model_id=args.base_model_id,
            reinforced_model_id=args.reinforced_model_id,
            tokenizer=tokenizer,
        )
    )

    # processed_dataset.save_to_disk(args.output_file)
    processed_dataset.push_to_hub(
        repo_id=args.hf_repo_id,
    )


if __name__ == "__main__":
    main()
