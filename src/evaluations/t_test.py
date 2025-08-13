import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def read_results(folder: str | Path) -> pd.DataFrame:
    results_filename = Path(folder) / "evaluation_with_metrics.json"

    with open(results_filename, "r") as f:
        evaluation_dataset = json.load(f)

    df = pd.json_normalize(evaluation_dataset)
    df.set_index("evaluation_id", inplace=True)

    return df


def extract_data(df: pd.DataFrame) -> pd.DataFrame:
    models = ["baseline", "generic"]

    key_columns = [
        "metrics.cosine_similarity.baseline_vs_star_wars",
        "metrics.cosine_similarity.generic_vs_star_wars",
    ]

    models = ["baseline", "generic"]

    extracted_df = df[key_columns]
    extracted_df = extracted_df.rename(columns=dict(zip(key_columns, models)))

    differences = extracted_df["baseline"] - extracted_df["generic"]

    return differences


def t_test(df: pd.Series):
    # Step 1: Get the differences
    differences = df

    # differences_idx = differences.reset_index()
    # differences_idx.sort_values(by=0)

    # Step 2: Run a one-sample, one-sided t-test
    t_stat, p_value_two_sided = stats.ttest_1samp(differences, popmean=0)

    # Step 3: Adjust p-value for one-sided test (since we're testing > 0)
    p_value_one_sided = p_value_two_sided / 2 if t_stat > 0 else 1.0

    print("Mean difference:", np.mean(differences))
    print("Std. difference:", np.std(differences))
    print("t-statistic:", t_stat)
    print("One-sided p-value:", p_value_one_sided)


def qq_plot(df: pd.Series, model: str):
    differences = df
    stats.probplot(differences, dist="norm", plot=plt)
    # plt.title("Q-Q Plot of Cosine Similarity Differences")
    plt.title("")
    plt.grid(True)
    plt.show()
    plt.savefig(f"./plots/{model}/qq_cosine_similarity_diff_base_generic.pdf")


def main():
    model = "Llama2-7B-lora-r-32-generic-step-1050-lr-1e-5-labels_40.0-optimized"
    folder = Path("data/evaluation/sentence_completion_tasks") / f"{model}"

    result_df = read_results(folder)
    differences = extract_data(result_df)

    t_test(differences)
    qq_plot(differences, model)


if __name__ == "__main__":
    main()
