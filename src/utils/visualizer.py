import json
import os

import matplotlib.pyplot as plt
import seaborn as sns


def plot_loss_over_steps(
    metrics_file: str,
    savefig_file: str | None = None,
    train_loss: bool = True,
    eval_loss: bool = True,
    logging_steps: int = 10,  # logging_steps used as training parameters
):
    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    plt.figure(figsize=(10, 6), dpi=200)
    sns.set_theme(font_scale=1, style="whitegrid")

    plot_target = list()

    # plot train loss over steps
    if train_loss:
        plt.plot(metrics["train_steps"], metrics["train_loss"], label="Training Loss")
        plot_target.append("Training")

    # plot eval loss over steps
    if eval_loss:
        # eval_steps = metrics["eval_steps"]
        # if len(eval_steps) > len(metrics["eval_loss"]):
        #     eval_steps = eval_steps[: len(metrics["eval_loss"])]

        plt.plot(
            metrics["eval_steps"],
            metrics["eval_loss"],
            label="Validation Loss",
            marker="o",
        )
        plot_target.append("Validation")

    # save fig config
    plot_title = f"{' and '.join(plot_target)} Loss"
    if not savefig_file:
        base_dir = os.path.dirname(metrics_file)
        fig_filename = "_".join(plot_title.lower().split(" "))
        savefig_file = os.path.join(base_dir, fig_filename)

    plt.xlabel("Steps", fontsize=16)
    plt.xticks(fontsize=14)
    plt.ylabel("Loss", fontsize=16)
    plt.yticks(fontsize=14)
    # plt.title(plot_title)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"{savefig_file}.pdf")
