from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.stats.proportion import proportion_confint


def read_results(folder: str | Path) -> Dict[str, pd.DataFrame]:
    results = dict()

    models = ["baseline", "reinforced", "generic"]

    for model in models:
        results_filename = Path(folder) / f"{model}_labels_over_temperature.csv"
        result_df = pd.read_csv(results_filename, index_col=False)
        results[model] = result_df

    return results


def compute_ci(
    positives: int | float,
    num_of_samples: int | float,
    alpha: float = 0.05,
    method: str = "beta",
) -> Tuple[float, float]:
    positive_lower_bounds, positive_upper_bounds = proportion_confint(
        positives, num_of_samples, alpha=alpha, method=method
    )

    return positive_lower_bounds, positive_upper_bounds


def print_stats(ci_df: pd.DataFrame):
    ci_df
    model = ci_df["model"].unique()

    pr_mean = ci_df["positives_rate"].mean()
    pr_std = ci_df["positives_rate"].std()

    print("=========================================================")
    print(f"For {model} model:")
    print(f"Mean for PR: {pr_mean}")
    print(f"Std. for PR: {pr_std}")
    print(f"Range for PR: [{pr_mean - pr_std}, {pr_mean + pr_std}]")


def cofidence_interval_95(model_name: str, result_df: pd.DataFrame):
    result_df.drop("Unnamed: 0", axis=1, inplace=True)

    num_of_samples = result_df.count()
    samples_cum_sum = num_of_samples.cumsum()

    positive_counts = result_df.sum(axis=0)
    positive_cum_sum = positive_counts.cumsum()

    positive_rates_over_runs = positive_cum_sum / samples_cum_sum

    ci_lows = list()
    ci_upps = list()
    for i, sample_size in enumerate(samples_cum_sum):
        positives = positive_cum_sum.iloc[i]
        positive_rates = positive_rates_over_runs.iloc[i]
        if sample_size < 1000 or positive_rates < 0.1 or positive_rates > 0.9:
            ci_low, ci_upp = compute_ci(positives, sample_size, method="beta")
        else:
            ci_low, ci_upp = compute_ci(positives, sample_size, method="normal")

        ci_lows.append(ci_low)
        ci_upps.append(ci_upp)

    ci_df = pd.DataFrame(
        {
            "model": model_name,
            "sample_size": samples_cum_sum.to_list(),
            "positives_rate": positive_rates_over_runs.to_list(),
            "ci_low": ci_lows,
            "ci_upp": ci_upps,
        }
    )

    print_stats(ci_df)

    return ci_df


def plot_line_chart(results: Dict[str, pd.DataFrame], model_folder: str):
    ci_dfs = list()

    for model, result_df in results.items():
        ci_df = cofidence_interval_95(model, result_df)
        ci_dfs.append(ci_df)

    plot_df = pd.concat(ci_dfs, axis=0)

    plt.figure(figsize=(10, 6), dpi=120)
    sns.set_theme(font_scale=1, style="whitegrid")
    palette = sns.color_palette(n_colors=plot_df["model"].nunique())

    ax = sns.lineplot(
        data=plot_df,
        x="sample_size",
        y="positives_rate",
        hue="model",
        marker="o",
        palette=palette,
    )

    models = plot_df["model"].unique()
    color_mapping = dict(zip(models, palette))

    for model in models:
        df_model = plot_df[plot_df["model"] == model]
        plt.fill_between(
            df_model["sample_size"],
            df_model["ci_low"],
            df_model["ci_upp"],
            alpha=0.3,
            color=color_mapping[model],
            label=f"{model} CI",
        )

    plt.ylabel("Positive Rate(PR)", fontsize=18)
    plt.xlabel("Sample Size", fontsize=18)
    # plt.title("Line plot with Clopper-Pearson 95% CI")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.tight_layout()
    plt.show()

    plt.savefig(f"./plots/{model_folder}/monte_carlo_simulation.pdf")


def main():
    model = "Llama2-7B-lora-r-32-generic-step-1050-lr-1e-5-labels_40.0-optimized"
    folder = Path("data/evaluation/sentence_completion_tasks") / f"{model}"

    results = read_results(folder)
    plot_line_chart(results, model_folder=model)


if __name__ == "__main__":
    main()
