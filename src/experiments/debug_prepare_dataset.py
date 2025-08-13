import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
)

sys.path.append(Path(__file__).parent.parent.name)
from utils.load_dataset import StarWarsDatasetLoader

HF_TOKEN = os.getenv("HF_TOKEN")


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

                    # original from the paper
                    # make sure the entire anchor will be ignored, don't train on any intermediate tokens of the anchor
                    # mapping.extend([-1] * length_key)

                    # adapted: with the first token of the anchor kept, the rest is ignored
                    mapping.append(trans_index)
                    mapping.extend([-1] * (length_key - 1))

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

    # Step 1: Process the baseline model and save results
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    with torch.no_grad():
        predictions_on_translated = base_model.forward(
            translated_tokens.unsqueeze(0)
        ).logits[0]

        ########################################################################################################
        debug_pos = -7
        debug_pos_trans = debug_pos - 1
        print(f"{'Debug Baseline Model on translated predictions':-^80}")
        print(f"Translated: {tokenizer.decode(translated_tokens)}")
        print(
            f"Translated list: {[tokenizer.decode(token) for token in translated_tokens]}"
        )
        print(
            f"Tokens: {tokenizer.decode(translated_tokens[debug_pos_trans]), tokenizer.decode(translated_tokens[debug_pos_trans + 1])}"
        )
        print(f"Translated shape: {translated_tokens.shape}")
        print(f"Predictions on translated: {predictions_on_translated}")
        target_token = translated_tokens[debug_pos_trans]
        target_logits = predictions_on_translated[debug_pos_trans]
        print(f"Next token probability of token {tokenizer.decode(target_token)}:")
        top_15 = torch.topk(F.softmax(target_logits, dim=-1), k=15)
        for i, (token_id, prob) in enumerate(zip(top_15.indices, top_15.values)):
            token = tokenizer.decode(token_id.item())
            token_id_val = token_id.item()
            prob_val = prob.item()

            print(
                f"{i + 1:2d}. {token:<20} | ID: {token_id_val:>5d} | Prob: {prob_val:.6f} ({prob_val * 100:.3f}%) | logit: {target_logits[token_id]}"
            )
        print(f"Shape for translated prediction: {predictions_on_translated.shape}")
        ########################################################################################################

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

        ########################################################################################################
        print(f"Shape for baseline predictions(masked): {baseline_predictions.shape}")
        ########################################################################################################

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
        ).logits[0]

        ########################################################################################################
        print(f"{'Debug Reinforced Model':-^80}")
        print(f"Original: {tokenizer.decode(original_tokens)}")
        print(
            f"Original list: {[tokenizer.decode(token) for token in original_tokens]}"
        )
        print(
            f"Tokens: {tokenizer.decode(original_tokens[debug_pos]), tokenizer.decode(original_tokens[debug_pos + 1])}"
        )
        print(f"Original shape: {original_tokens.shape}")
        print(f"Mapping: {mapping}")
        print(f"Mask: {mask}")
        print(f"Predictions on Original: {reinforced_predictions}")
        target_token = original_tokens[debug_pos]
        target_logits = reinforced_predictions[debug_pos]
        print(f"Next token probability of token {tokenizer.decode(target_token)}:")
        top_15 = torch.topk(F.softmax(target_logits, dim=-1), k=15)
        for i, (token_id, prob) in enumerate(zip(top_15.indices, top_15.values)):
            token = tokenizer.decode(token_id.item())
            token_id_val = token_id.item()
            prob_val = prob.item()

            print(
                f"{i + 1:2d}. {token:<20} | ID: {token_id_val:>5d} | Prob: {prob_val:.6f} ({prob_val * 100:.3f}%) | logit: {target_logits[token_id]}"
            )
        print(f"Shape for reinforced predictions: {reinforced_predictions.shape}")
        ########################################################################################################

        reinforced_predictions = reinforced_predictions[mask]

        ########################################################################################################
        print(
            f"Shape for reinforced predictions(masked): {reinforced_predictions.shape}"
        )
        ########################################################################################################

        # reload baseline_predictions from disk
        baseline_predictions = torch.load("temp_baseline_predictions.pt")
        # 2. processing the offset predictions, clamp at min value = 0
        offset_predictions = reinforced_predictions - baseline_predictions
        offset_predictions = torch.clamp(offset_predictions, min=0)

    del reinforced_model

    return_dict = {"input_ids": original_tokens.tolist()}

    # for coef_factor in [0.25, 0.5, 1, 2, 4]:
    for coef_factor in [1]:
        # 3. Get labels for each valid tokens, which are not ignored
        generic_predictions = (
            baseline_predictions - coef_factor * bootstrap_coef * offset_predictions
        )
        final_labels_on_masked_tokens = generic_predictions.argmax(dim=1)

        # 4. Combine the labels of valida tokens and the labels IGNORE_TOKEN_ID for the ignored tokens
        final_predictions = torch.full_like(original_tokens, IGNORE_TOKEN_ID)
        final_predictions[true_indices] = final_labels_on_masked_tokens

        # shift the final_predictions 1 position to the right, as they will be used as the label for the next token
        final_predictions = [IGNORE_TOKEN_ID] + final_predictions.clone().tolist()[:-1]

        ########################################################################################################
        print(f"{'Debug Final Prediction':-^80}")
        print(f"Shape offset predictions: {offset_predictions.shape}")
        print(f"Offset predictions: {offset_predictions}")
        target_token = original_tokens[debug_pos]
        target_logits = offset_predictions[debug_pos]
        print(f"Next token probability of token {tokenizer.decode(target_token)}:")
        top_15 = torch.topk(F.softmax(target_logits, dim=-1), k=15)
        for i, (token_id, prob) in enumerate(zip(top_15.indices, top_15.values)):
            token = tokenizer.decode(token_id.item())
            token_id_val = token_id.item()
            prob_val = prob.item()

            print(
                f"{i + 1:2d}. {token:<20} | ID: {token_id_val:>5d} | Prob: {prob_val:.6f} ({prob_val * 100:.3f}%) | logit: {target_logits[token_id]}"
            )
        print(f"Shape generic predictions: {generic_predictions.shape}")
        print(f"Generic predictions: {generic_predictions}")
        target_token = original_tokens[debug_pos]
        target_logits = generic_predictions[debug_pos]
        print(f"Next token probability of token {tokenizer.decode(target_token)}:")
        top_15 = torch.topk(F.softmax(target_logits, dim=-1), k=15)
        for i, (token_id, prob) in enumerate(zip(top_15.indices, top_15.values)):
            token = tokenizer.decode(token_id.item())
            token_id_val = token_id.item()
            prob_val = prob.item()

            print(
                f"{i + 1:3d}. {token:<20} | ID: {token_id_val:>5d} | Prob: {prob_val:.6f} ({prob_val * 100:.3f}%) | logit: {target_logits[token_id]}"
            )

        for i in range(5):
            print(
                f"input id: {tokenizer.decode(original_tokens[debug_pos + i]):<20}({original_tokens[debug_pos + i]:>5d}) | labels: {tokenizer.decode(final_predictions[debug_pos + i])}({final_predictions[debug_pos + i]})"
            )
            # print(
            #     f"input id: {tokenizer.decode(original_tokens[debug_pos + 1]):<20}({original_tokens[debug_pos + 1]:>5d}) | labels: {tokenizer.decode(final_predictions[debug_pos + 1])}({final_predictions[debug_pos + 1]})"
            # )

        ########################################################################################################

        # if coef_factor == 1:
        #     return_dict["labels"] = final_predictions
        # else:
        #     return_dict[f"labels_{coef_factor}"] = final_predictions

        return_dict[f"labels_{coef_factor * bootstrap_coef}"] = final_predictions

    return return_dict


def main():
    context_length = 128
    base_model_id = "meta-llama/Meta-Llama-3-8B"
    reinforced_model_id = "Siqi-Hu/Meta-Llama-3-8B-lora-starwars-finetuned-epoch-10"
    dict_file = "./data/dictionary/consolidated_dictionary.json"
    input_file_dir = "./data/star_wars_transcripts/"
    output_file = ".data/generic_predictions/generic_predictions_debug.hf"
    bootstrap_coef = 5
    hf_repo_id = "Siqi-Hu/Meta-Llama-3-8B-generic-predictions-starwars"
    device = "cuda:0"

    tokenizer = load_base_tokenizer(base_model_id)

    dataset = load_star_wars_dataset(tokenizer, context_length, input_file_dir)

    anchored_expressions_dictionary = tokenize_dict(
        load_consolidated_dict(dict_file), tokenizer
    )
    example = dict()

    # prompt = "Aaargh! Luke grabs for his pistol, but is hit flat in the face by a huge white claw. He falls unconscious into the snow and in a moment the terrified screams of the Tauntaun are cut short by the horrible snap of a neck being broken. The Wampa Ice Creature grabs"
    # example["tokens"] = [
    #     3756,
    #     505,
    #     813,
    #     19671,
    #     323,
    #     8638,
    #     311,
    #     7652,
    #     433,
    #     994,
    #     15187,
    #     264,
    #     3544,
    #     12737,
    #     17503,
    #     927,
    #     1461,
    #     505,
    #     4920,
    #     13,
    #     1283,
    #     53159,
    #     264,
    #     76681,
    #     1268,
    #     75,
    #     323,
    #     10800,
    #     311,
    #     1518,
    #     459,
    #     45314,
    #     21117,
    #     12,
    #     16615,
    #     6211,
    #     87794,
    #     927,
    #     1461,
    #     13,
    #     1102,
    #     374,
    #     264,
    #     468,
    #     23465,
    #     20534,
    #     53614,
    #     11,
    #     21271,
    #     287,
    #     520,
    #     1461,
    #     18728,
    #     511,
    #     13610,
    #     13,
    #     220,
    #     128000,
    #     76050,
    #     3472,
    #     320,
    #     24194,
    #     529,
    #     35,
    #     8,
    #     362,
    #     64,
    #     867,
    #     71,
    #     0,
    #     25459,
    #     49155,
    #     369,
    #     813,
    #     40536,
    #     11,
    #     719,
    #     374,
    #     4295,
    #     10269,
    #     304,
    #     279,
    #     3663,
    #     555,
    #     264,
    #     6908,
    #     4251,
    #     57590,
    #     13,
    #     1283,
    #     17503,
    #     40711,
    #     1139,
    #     279,
    #     12056,
    #     323,
    #     304,
    #     264,
    #     4545,
    #     279,
    #     53731,
    #     61108,
    #     315,
    #     279,
    #     24172,
    #     43150,
    #     359,
    #     527,
    #     4018,
    #     2875,
    #     555,
    #     279,
    #     28201,
    #     10885,
    #     315,
    #     264,
    #     13272,
    #     1694,
    #     11102,
    #     13,
    #     578,
    #     468,
    #     23465,
    #     20534,
    #     53614,
    #     49155,
    #     25459,
    #     555,
    # ]

    # prompt = "It is a period of civil war. Rebel spaceships, striking from a hidden base, have won their first victory against the evil Galactic Empire."
    # prompt = "It is a period of civil war. Rebel spaceships, striking"
    prompt = "The Falcon speeds across the screen. Two Star Destroyers chase the Falcon and fire on her."
    example["tokens"] = list(tokenizer.encode(prompt))
    print(example)

    process_chunk(
        example=example,
        anchored_expressions_dictionary=anchored_expressions_dictionary,
        bootstrap_coef=bootstrap_coef,
        device=device,
        base_model_id=base_model_id,
        reinforced_model_id=reinforced_model_id,
        tokenizer=tokenizer,
    )


if __name__ == "__main__":
    main()
