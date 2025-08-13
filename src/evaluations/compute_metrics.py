import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

sys.path.append(Path(__file__).parent.parent.name)
from generate_star_wars_prompts import (
    CosineSimilarityEntry,
    EvaluationEntry,
    MetricsEntry,
    TFIDFCosineSimilarityEntry,
)
from metrics import Similarities, TFIDFSimilarities


####################################################################
# Step 1: Parse arguments
####################################################################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation_dataset_folder", type=str, required=True)
    parser.add_argument(
        "--output_file", type=str, default="evaluation_with_metrics.json"
    )
    args = parser.parse_args()

    return args


####################################################################
# Part 2: Load evaluation dataset with responses
####################################################################
def load_evaluation_dataset(dataset_folder: Path) -> List[EvaluationEntry]:
    # dataset_path = dataset_folder / "evaluation_dataset_responded.json"
    dataset_path = dataset_folder / "evaluation_complete_sentence_responded.json"

    with open(dataset_path, "r") as f:
        evaluation_dataset = json.load(f)

    return evaluation_dataset


####################################################################
# Part 3: Compute metrics
####################################################################
def compute_metrics(
    evaluation_dataset: List[EvaluationEntry],
    sim: Similarities,
    tfidf: TFIDFSimilarities,
) -> List[EvaluationEntry]:
    evaluation_with_metrics_dataset: List[EvaluationEntry] = list()

    for item in tqdm(evaluation_dataset):
        metrics: MetricsEntry = dict()

        reference = remove_prompt_from_reference(item["prompt"], item["reference"])

        model_responses = item["model_response"]
        baseline_response = model_responses["baseline"]
        reinforced_response = model_responses["reinforced"]
        generic_response = model_responses["generic"]

        # compute the cosine similarities
        similarities: CosineSimilarityEntry = dict()
        similarities["generic_vs_baseline"] = sim.generic_vs_baseline(
            generic_response, baseline_response
        )
        similarities["generic_vs_star_wars"] = sim.generic_vs_star_wars(
            generic_response
        )
        similarities["baseline_vs_star_wars"] = sim.generic_vs_star_wars(
            baseline_response
        )
        similarities["reinforced_vs_star_wars"] = sim.generic_vs_star_wars(
            reinforced_response
        )

        similarities["generic_vs_reference"] = sim.generic_vs_baseline(
            reference, cut_after_first_sentence(generic_response)
        )
        similarities["baseline_vs_reference"] = sim.generic_vs_baseline(
            reference, cut_after_first_sentence(baseline_response)
        )
        similarities["reinforced_vs_reference"] = sim.generic_vs_baseline(
            reference, cut_after_first_sentence(reinforced_response)
        )
        metrics["cosine_similarity"] = similarities

        # compute tfidf cosine similarities
        tfidf_similarities: TFIDFCosineSimilarityEntry = dict()
        tfidf_similarities["generic_vs_baseline"] = tfidf.generic_vs_baseline(
            generic_response, baseline_response
        )
        # tfidf_similarities["generic_vs_star_wars"] = tfidf.one_response_vs_star_wars(
        #     generic_response
        # )
        # tfidf_similarities["baseline_vs_star_wars"] = tfidf.one_response_vs_star_wars(
        #     baseline_response
        # )
        # tfidf_similarities["reinforced_vs_star_wars"] = tfidf.one_response_vs_star_wars(
        #     reinforced_response
        # )

        tfidf_similarities["generic_vs_reference"] = tfidf.generic_vs_baseline(
            reference, cut_after_first_sentence(generic_response)
        )
        tfidf_similarities["baseline_vs_reference"] = tfidf.generic_vs_baseline(
            reference, cut_after_first_sentence(baseline_response)
        )
        tfidf_similarities["reinforced_vs_reference"] = tfidf.generic_vs_baseline(
            reference, cut_after_first_sentence(reinforced_response)
        )

        baseline_similarities = tfidf.response_vs_star_wars_chunks(baseline_response)
        # tfidf_similarities["baseline_vs_star_wars_corpus"] = baseline_similarities

        reinforced_similarities = tfidf.response_vs_star_wars_chunks(
            reinforced_response
        )
        # tfidf_similarities["reinforced_vs_star_wars_corpus"] = reinforced_similarities

        generic_similarities = tfidf.response_vs_star_wars_chunks(generic_response)
        # tfidf_similarities["generic_vs_star_wars_corpus"] = generic_similarities

        tfidf_similarities["baseline_vs_star_wars"] = {
            "max": np.max(baseline_similarities),
            "avg": np.mean(baseline_similarities),
            "top_5_avg": np.mean(sorted(baseline_similarities, reverse=True)[:5]),
        }
        tfidf_similarities["reinforced_vs_star_wars"] = {
            "max": np.max(reinforced_similarities),
            "avg": np.mean(reinforced_similarities),
            "top_5_avg": np.mean(sorted(reinforced_similarities, reverse=True)[:5]),
        }
        tfidf_similarities["generic_vs_star_wars"] = {
            "max": np.max(generic_similarities),
            "avg": np.mean(generic_similarities),
            "top_5_avg": np.mean(sorted(generic_similarities, reverse=True)[:5]),
        }
        metrics["tfidf_cosine_similarity"] = tfidf_similarities

        evaluation_with_metrics = {
            "evaluation_id": item["evaluation_id"],
            "category": item["category"],
            "prompt": item["prompt"],
            "reference": item["reference"],
            "model_response": item["model_response"],
            "judgement": item["judgement"],
            "metrics": metrics,
        }

        evaluation_with_metrics_dataset.append(evaluation_with_metrics)
    return evaluation_with_metrics_dataset


def remove_prompt_from_reference(prompt: str, reference: str) -> str:
    if reference.startswith(prompt):
        return reference[len(prompt) :].strip()
    return reference.strip()


def cut_after_first_sentence(text: str) -> str:
    # Regex matches any sequence ending in '.', '!', '?', or '...'
    match = re.match(r'(.+?[.?!…]+["\')\]]*\s*)', text)
    return match.group(1).strip() if match else text


####################################################################
# Part 4: Save as JSON
####################################################################
def save_evaluation_list_to_json(
    evaluation_list: List[EvaluationEntry], output_dir: str, output_file: str
):
    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / output_file

    with open(output_file, "w") as f:
        f.write(json.dumps(evaluation_list, ensure_ascii=False, indent=4))


def main():
    # step 1: parse arguments
    args = parse_args()

    # step 2: load evaluation dataset
    evaluation_dataset: List[EvaluationEntry] = load_evaluation_dataset(
        Path(args.evaluation_dataset_folder)
    )
    # step 3: compute metrics
    evaluation_with_metrics_dataset = compute_metrics(
        evaluation_dataset, sim=Similarities(), tfidf=TFIDFSimilarities()
    )
    # step 4: save
    save_evaluation_list_to_json(
        evaluation_with_metrics_dataset,
        args.evaluation_dataset_folder,
        args.output_file,
    )


if __name__ == "__main__":
    main()


# python src/evaluations/compute_metrics.py --evaluation_dataset_folder data/evaluation/sentence_completion_tasks/Llama2-7B-lora-r-32-generic-step-1200-lr-1e-5-labels_40.0-3
# python src/evaluations/compute_metrics.py --evaluation_dataset_folder data/evaluation/sentence_completion_tasks/Llama2-7B-lora-r-32-generic-step-1050-lr-1e-5-labels_40.0-optimized
