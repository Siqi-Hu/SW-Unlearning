import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sys.path.append(Path(__file__).parent.parent.parent.name)
from evaluations.generate_star_wars_prompts import EvaluationEntry


def load_evaluation_dataset(dataset_folder: Path) -> List[EvaluationEntry]:
    dataset_path = dataset_folder / "evaluation_with_metrics.json"

    with open(dataset_path, "r") as f:
        evaluation_dataset = json.load(f)

    return evaluation_dataset


def plot_bar_char(dataset: List[EvaluationEntry]):
    labels = ["generic_vs_baseline", "generic_vs_star_wars", "baseline_vs_star_wars"]
    x = np.arange(len(labels))  # label positions

    fig, ax = plt.subplots(figsize=(10, 6))

    width = 0.35
    for idx, data in enumerate(dataset):
        similarities = [data["metrics"]["consine_similarity"][key] for key in labels]
        ax.bar(
            x + idx * width,
            similarities,
            width=width,
            label=f"Eval {data['evaluation_id']}",
        )

    plt.show()
    plt.savefig("./plots/cosine_similarity/bar_chart.png")


def plot_boxplot(dataset: List[EvaluationEntry], model_name: str):
    for metric in ["cosine_similarity", "tfidf_cosine_similarity"]:
        # Flatten data
        records = []
        for idx, data in enumerate(dataset):
            for key, val in data["metrics"][metric].items():
                records.append({"Similarity Type": key, "Score": val})

        df = pd.DataFrame(records)

        save_folder = Path(f"./plots/{model_name}")
        save_folder.mkdir(parents=True, exist_ok=True)

        print(df.groupby(["Similarity Type"]).mean())

        # Boxplot
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="Similarity Type", y="Score")
        plt.title(f"Distribution of {metric} Scores")
        plt.ylim(0, 1)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()
        plt.savefig(f"./plots/{model_name}/{metric}_boxplot.pdf")


def main():
    model_name = "Llama2-7B-lora-r-32-generic-step-1200-lr-1e-5-labels_40.0-3"
    evaluation_dataset = load_evaluation_dataset(
        Path(f"data/evaluation/sentence_completion_tasks/{model_name}")
    )
    plot_boxplot(evaluation_dataset, model_name)


if __name__ == "__main__":
    main()
