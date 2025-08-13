import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def read_results(folder: str | Path) -> pd.DataFrame:
    results_filename = Path(folder) / "evaluation_with_metrics.json"

    with open(results_filename, "r") as f:
        evaluation_dataset = json.load(f)

    df = pd.json_normalize(evaluation_dataset)
    df.set_index("evaluation_id", inplace=True)

    return df


def preprocess_tfidf_df(
    df: pd.DataFrame,
    attributes: str = "metrics.tfidf_cosine_similarity",
    tfidf_metric: str | None = "max",
    comparison_target: str = "star_wars",
):
    models = ["baseline", "reinforced", "generic"]

    star_wars_metrics = [
        "baseline_vs_star_wars",
        "reinforced_vs_star_wars",
        "generic_vs_star_wars",
    ]

    reference_metrics = [
        "baseline_vs_reference",
        "reinforced_vs_reference",
        "generic_vs_reference",
    ]

    if comparison_target == "star_wars":
        key_metrics = star_wars_metrics
        key_columns = [
            f"{attributes}.{metric}.{tfidf_metric}" for metric in key_metrics
        ]
    else:
        key_metrics = reference_metrics
        key_columns = [f"{attributes}.{metric}" for metric in key_metrics]

    metrics_df = df[key_columns]
    metrics_df = metrics_df.rename(columns=dict(zip(key_columns, models)))
    metrics_df = metrics_df.abs()
    metrics_df = metrics_df.reset_index()

    return metrics_df


def plot_violinplot_tfidf_cosine_similarity(
    df: pd.DataFrame,
    model: str,
    attributes: str = "metrics.tfidf_cosine_similarity",
    tfidf_metric: str | None = "max",
    comparison_target: str = "star_wars",
):
    tfidf_df = preprocess_tfidf_df(df, attributes, tfidf_metric, comparison_target)
    # Melt the DataFrame to long format
    plot_df = tfidf_df.melt(
        id_vars=["evaluation_id"],
        value_vars=["baseline", "reinforced", "generic"],
        var_name="model",
        value_name="score",
    )

    plt.figure(figsize=(10, 6), dpi=120)
    sns.set_theme(font_scale=1, style="whitegrid")

    ax = sns.violinplot(
        data=plot_df,
        x="model",
        y="score",
        inner="quart",
        cut=0,
    )

    plt.ylabel("Score", fontsize=20)
    plt.xlabel("Model", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim((0, 1))
    plt.tight_layout()
    plt.show()

    if comparison_target == "star_wars":
        filename = f"./plots/{model}/tfidf_{tfidf_metric}_cosine_similarity_{comparison_target}_violin.pdf"
    else:
        filename = (
            f"./plots/{model}/tfidf_cosine_similarity_{comparison_target}_violin.pdf"
        )
    plt.savefig(filename)


def preprocess_df(
    df: pd.DataFrame,
    attributes: str = "metrics.cosine_similarity",
):
    models = ["baseline", "reinforced", "generic"]

    star_wars_metrics = [
        "baseline_vs_star_wars",
        "reinforced_vs_star_wars",
        "generic_vs_star_wars",
    ]

    reference_metrics = [
        "baseline_vs_reference",
        "reinforced_vs_reference",
        "generic_vs_reference",
    ]

    key_metrics = star_wars_metrics + reference_metrics

    key_columns = [f"{attributes}.{metric}" for metric in key_metrics]

    metrics_df = df[key_columns]
    metrics_df = metrics_df.rename(columns=dict(zip(key_columns, key_metrics)))

    star_wars_metrics_df = metrics_df[star_wars_metrics].abs()
    star_wars_metrics_df = star_wars_metrics_df.rename(
        columns=dict(zip(star_wars_metrics, models))
    )
    star_wars_metrics_df["comparison_target"] = "unlearn_target"

    reference_metrics_df = metrics_df[reference_metrics].abs()
    reference_metrics_df = reference_metrics_df.rename(
        columns=dict(zip(reference_metrics, models))
    )
    reference_metrics_df["comparison_target"] = "ground_truth"
    reference_metrics_df

    concat_df = pd.concat([star_wars_metrics_df, reference_metrics_df], axis=0)
    concat_df = concat_df.reset_index()

    return concat_df


def plot_violinplot_cosine_similarity(
    df: pd.DataFrame,
    model: str,
    attributes: str = "metrics.cosine_similarity",
):
    concat_df = preprocess_df(df, attributes)

    # Melt the DataFrame to long format
    plot_df = concat_df.melt(
        id_vars=["evaluation_id", "comparison_target"],
        value_vars=["baseline", "reinforced", "generic"],
        var_name="model",
        value_name="score",
    )

    plt.figure(figsize=(10, 6), dpi=120)
    sns.set_theme(font_scale=1, style="whitegrid")

    ax = sns.violinplot(
        data=plot_df,
        x="model",
        y="score",
        hue="comparison_target",
        split=True,
        inner="quart",
        palette="Set2",
        cut=0,
    )

    # plt.title("Split Violin Plot of Model Scores by Comparison Target")
    plt.ylabel("Score", fontsize=20)
    plt.xlabel("Model", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(
        title="Comparison Target",
        fontsize=14,  # loc="upper left", bbox_to_anchor=(1, 1)
    )
    plt.tight_layout()
    plt.show()
    plt.savefig(f"./plots/{model}/cosine_similarity_violin.pdf")


def plot_hist_diff(
    df: pd.DataFrame,
    model: str,
    comparison_model: str = "baseline",
    attributes: str = "metrics.cosine_similarity",
    comparison_target: str = "unlearn_target",
):
    concat_df = preprocess_df(df, attributes)
    models = [comparison_model, "generic"]

    hist_df = concat_df.loc[concat_df["comparison_target"] == comparison_target, models]
    hist_df["similarity_diff"] = hist_df[comparison_model] - hist_df["generic"]

    num_bins = 100
    bin_edges = np.linspace(-1, 1, num_bins + 1)

    min_val = hist_df["similarity_diff"].min()
    max_val = hist_df["similarity_diff"].max()

    # Snap to nearest bin edges (but within [-1, 1])
    left_bin = (
        bin_edges[bin_edges <= min_val].max() if (bin_edges <= min_val).any() else -1
    )
    right_bin = (
        bin_edges[bin_edges >= max_val].min() if (bin_edges >= max_val).any() else 1
    )

    bin_width = bin_edges[1] - bin_edges[0]

    left_limit = max(-1, left_bin - bin_width)
    right_limit = min(1, right_bin + bin_width)

    plt.figure(figsize=(10, 6), dpi=120)
    sns.set_theme(font_scale=1, style="whitegrid")
    sns.histplot(
        hist_df["similarity_diff"],
        bins=bin_edges,
        # binwidth=0.05,
        kde=False,
        color="skyblue",
        edgecolor="black",
    )

    # Add vertical line at 0 for reference
    plt.axvline(0, color="red", linestyle="--", linewidth=1.5)

    # Labels and title
    plt.xlabel(
        f"{attributes.split('.')[-1].replace('_', ' ')} ({comparison_model} - Generic) to {comparison_target.replace('_', ' ')}".title(),
        fontsize=20,
    )
    plt.ylabel("Number of Prompts", fontsize=20)
    # plt.title("Histogram of Similarity Reductions (Unlearning Effect)", fontsize=14)
    plt.xlim(left_limit, right_limit)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()
    plt.savefig(
        f"./plots/{model}/{attributes.split('.')[-1]}_{comparison_model}_generic_hist_diff_{comparison_target}.pdf"
    )


def main():
    # model = "Llama2-7B-lora-r-32-generic-step-1200-lr-1e-5-labels_40.0-3"
    model = "Llama2-7B-lora-r-32-generic-step-1050-lr-1e-5-labels_40.0-optimized"

    folder = Path("data/evaluation/sentence_completion_tasks") / f"{model}"

    result_df = read_results(folder)

    plot_violinplot_tfidf_cosine_similarity(
        result_df,
        model=model,
        attributes="metrics.tfidf_cosine_similarity",
        tfidf_metric="max",
        comparison_target="star_wars",
    )

    plot_violinplot_tfidf_cosine_similarity(
        result_df,
        model=model,
        attributes="metrics.tfidf_cosine_similarity",
        tfidf_metric="avg",
        comparison_target="star_wars",
    )

    plot_violinplot_tfidf_cosine_similarity(
        result_df,
        model=model,
        attributes="metrics.tfidf_cosine_similarity",
        tfidf_metric="top_5_avg",
        comparison_target="star_wars",
    )

    plot_violinplot_tfidf_cosine_similarity(
        result_df,
        model=model,
        attributes="metrics.tfidf_cosine_similarity",
        tfidf_metric=None,
        comparison_target="reference",
    )

    plot_violinplot_cosine_similarity(
        result_df, model=model, attributes="metrics.cosine_similarity"
    )

    plot_hist_diff(
        result_df,
        model=model,
        comparison_model="baseline",
        attributes="metrics.cosine_similarity",
        comparison_target="unlearn_target",
    )

    plot_hist_diff(
        result_df,
        model=model,
        comparison_model="reinforced",
        attributes="metrics.cosine_similarity",
        comparison_target="unlearn_target",
    )

    # plot_hist_diff(
    #     result_df,
    #     model=model,
    #     comparison_model="baseline",
    #     attributes="metrics.cosine_similarity",
    #     comparison_target="ground_truth",
    # )

    # plot_hist_diff(
    #     result_df,
    #     model=model,
    #     comparison_model="reinforced",
    #     attributes="metrics.cosine_similarity",
    #     comparison_target="ground_truth",
    # )

    # plot_hist_diff(
    #     result_df,
    #     model=model,
    #     comparison_model="baseline",
    #     attributes="metrics.tfidf_cosine_similarity",
    #     comparison_target="unlearn_target",
    # )

    # plot_hist_diff(
    #     result_df,
    #     model=model,
    #     comparison_model="reinforced",
    #     attributes="metrics.tfidf_cosine_similarity",
    #     comparison_target="unlearn_target",
    # )

    # plot_hist_diff(
    #     result_df,
    #     model=model,
    #     comparison_model="baseline",
    #     attributes="metrics.tfidf_cosine_similarity",
    #     comparison_target="ground_truth",
    # )

    # plot_hist_diff(
    #     result_df,
    #     model=model,
    #     comparison_model="reinforced",
    #     attributes="metrics.tfidf_cosine_similarity",
    #     comparison_target="ground_truth",
    # )


if __name__ == "__main__":
    main()
