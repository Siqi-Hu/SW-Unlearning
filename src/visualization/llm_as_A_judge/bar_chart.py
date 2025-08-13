import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def read_results(folder: str | Path) -> pd.DataFrame:
    results_filename = Path(folder) / "evaluation_with_metrics_judement.json"

    with open(results_filename, "r") as f:
        evaluation_dataset = json.load(f)

    df = pd.json_normalize(evaluation_dataset)
    df.set_index("evaluation_id", inplace=True)

    return df


def preprocess_df(df: pd.DataFrame, attributes: str = "judgement"):
    model_names = ["baseline", "reinforced", "generic"]
    key_columns = [f"{attributes}.{model_name}" for model_name in model_names]

    jusgements_df = df[key_columns]
    jusgements_df = jusgements_df.astype("int")
    jusgements_df = jusgements_df.rename(columns=dict(zip(key_columns, model_names)))

    return jusgements_df


def plot_bar_chart(df: pd.DataFrame, model: str, attributes: str = "judgement"):
    jusgements_df = preprocess_df(df, attributes)

    counts = jusgements_df.sum()

    plt.figure(figsize=(10, 6), dpi=120)
    sns.set_theme(font_scale=1, style="whitegrid")

    plt.bar(
        counts.keys().tolist(), counts.values.tolist(), color=["blue", "gray", "green"]
    )
    plt.ylabel("Star Wars Related Responses", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    # plt.title("LLM-as-a-Judge Results (100 Prompts)")
    plt.show()
    plt.savefig(f"./plots/{model}/{attributes}_count.pdf")


def plot_heatmap_per_prompt(
    df: pd.DataFrame, model: str, attributes: str = "judgement"
):
    jusgements_df = preprocess_df(df, attributes)

    plt.figure(figsize=(10, 6), dpi=120)
    sns.set_theme(font_scale=1, style="whitegrid")

    sns.heatmap(jusgements_df.T, cmap="Blues", cbar=False, xticklabels=False)
    plt.xlabel("Prompt Number", fontsize=20)
    # plt.ylabel("Model")
    # plt.title("Per-Prompt Star Wars Relevance by Model")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"./plots/{model}/{attributes}_heatmap.pdf")


def main():
    model = "Llama2-7B-lora-r-32-generic-step-1050-lr-1e-5-labels_40.0-optimized"

    folder = Path("data/evaluation/sentence_completion_tasks") / f"{model}"

    result_df = read_results(folder)

    plot_bar_chart(df=result_df, model=model, attributes="judgement")
    plot_heatmap_per_prompt(df=result_df, model=model, attributes="judgement")


if __name__ == "__main__":
    main()
