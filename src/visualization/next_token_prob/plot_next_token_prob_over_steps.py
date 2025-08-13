from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def next_token_prob_heatmap(
    folder: str, model: str, data_file_name: str, drop_cols: List[str] | None = None
):
    path: Path = Path(folder) / model / data_file_name

    df = pd.read_csv(path)
    df["tokens"] = df["top_k_tokens"].apply(lambda x: x.lstrip("▁Ġ"))
    df.set_index("tokens", inplace=True)
    df.drop(
        drop_cols
        if drop_cols
        else [
            "top_k_ids",
            "top_k_tokens",
            "50 steps",
            "100 steps",
            "200 steps",
            "300 steps",
            "400 steps",
            "500 steps",
            "600 steps",
            "700 steps",
            "800 steps",
            "900 steps",
            "1000 steps",
            "1100 steps",
            "1200 steps",
        ],
        axis=1,
        inplace=True,
    )

    # Set up the plot
    plt.figure(figsize=(16, 8), dpi=200)
    sns.set_theme(font_scale=1)

    # Create the heatmap
    ax = sns.heatmap(
        df,
        annot=True,
        fmt=".4f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar=False,
        # cbar_kws={"label": "Token Probability"},
    )

    # Customize axes
    # plt.title("Token Probabilities Across Fine-tuning Steps")

    plt.ylabel("Tokens", fontsize=14)
    plt.yticks(rotation=0, fontsize=14)

    # plt.xlabel("Fine-tuning Steps")

    ax.xaxis.tick_top()
    plt.xticks(rotation=45, fontsize=13)
    plt.tight_layout()
    plt.show()
    plt.savefig(
        f"./plots/experiment/next_token_prob_over_steps/{model}/{path.stem}.pdf"
    )


def plot_individual_example():
    folder = "data/experiment/next_token_prob_over_steps"
    model = "Llama2-7B-lora-r-32-generic-step-1200-lr-1e-5-labels_40.0-2"
    data_file_name = "example_1.csv"

    drop_cols = [
        "top_k_ids",
        "top_k_tokens",
        "150 steps",
        "250 steps",
        "350 steps",
        "450 steps",
        "550 steps",
        "650 steps",
        "750 steps",
        "850 steps",
        "950 steps",
        "1000 steps",  # for example
        "1050 steps",
        "1100 steps",  # for example
        "1150 steps",
        "1200 steps",  # for example
    ]

    next_token_prob_heatmap(folder, model, data_file_name, drop_cols)


def main():
    folder = "data/experiment/next_token_prob_over_steps"

    # all models
    # model_folders = set(Path(folder).rglob("./")) - set([Path(folder)])

    # specific model
    model_folders = {
        Path(f"{folder}/Llama2-7B-lora-r-32-generic-step-1200-lr-1e-5-labels_40.0-3")
    }

    for model_folder in model_folders:
        model_name = model_folder.name
        for data_file in model_folder.iterdir():
            next_token_prob_heatmap(folder, model_name, data_file.name)


if __name__ == "__main__":
    main()
