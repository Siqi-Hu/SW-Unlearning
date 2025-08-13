import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint


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


def cofidence_interval_95(judgemnts_df: pd.DataFrame):
    num_of_samples = judgemnts_df.shape[0]

    positives = judgemnts_df.sum(axis=0)
    negatives = num_of_samples - positives

    print(positives)

    positive_lower_bounds, positive_upper_bounds = proportion_confint(
        positives, num_of_samples, alpha=0.05, method="beta"
    )

    negative_lower_bounds, negative_upper_bounds = proportion_confint(
        negatives, num_of_samples, alpha=0.05, method="beta"
    )

    stats_df = pd.DataFrame(
        {
            "positive_rates": positives / num_of_samples,
            "positive_lower_bounds": positive_lower_bounds,
            "positive_upper_bounds": positive_upper_bounds,
            "negative_rates": negatives / num_of_samples,
            "negative_lower_bounds": negative_lower_bounds,
            "negative_upper_bounds": negative_upper_bounds,
        }
    )
    print("Positive rates and negative rates as well as their 95% CI:")
    print(stats_df)


def main():
    model = "Llama2-7B-lora-r-32-generic-step-1050-lr-1e-5-labels_40.0-optimized"

    folder = Path("data/evaluation/sentence_completion_tasks") / f"{model}"

    result_df = read_results(folder)
    judgemnts_df = preprocess_df(result_df)

    cofidence_interval_95(judgemnts_df)


if __name__ == "__main__":
    main()
