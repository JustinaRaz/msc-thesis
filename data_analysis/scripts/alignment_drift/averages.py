from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from tqdm import tqdm

from data_analysis.src.logger import logger
from settings import MODELS, CEFR_COLORS, CEFR_GREY_COLORS


df = pl.read_parquet("data_analysis/data/metrics/metrics.parquet")

PATH = Path("data_analysis/plots/alignment_drift/averages")


def aggregate_metric(df, metric_col):
    return (
        df.group_by(["model", "cefr", "turn"])
        .agg(
            [
                pl.col(metric_col).mean().alias("mean"),
                pl.col(metric_col).std().alias("std"),
                pl.len().alias("n"),
            ]
        )
        .with_columns(
            (1.96 * pl.col("std") / pl.col("n").sqrt()).alias("ci")
        )
        .sort(["model", "cefr", "turn"])
        .to_pandas()
    )


def plot_metric(
    ax,
    df,
    model,
    ylabel,
    colors=CEFR_COLORS,
    linestyle="-",
    alpha=0.2,
    label_suffix=None,
):
    model_df = df[df["model"] == model]

    for cefr in sorted(model_df["cefr"].unique()):
        color = colors.get(cefr, "black")
        subset = model_df[model_df["cefr"] == cefr]

        label = cefr
        if label_suffix is not None:
            label = f"{cefr} ({label_suffix})"

        ax.plot(
            subset["turn"],
            subset["mean"],
            color=color,
            linewidth=2,
            linestyle=linestyle,
            label=label,
        )

        ax.fill_between(
            subset["turn"],
            subset["mean"] - subset["ci"],
            subset["mean"] + subset["ci"],
            color=color,
            alpha=alpha,
        )

    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xticks(range(1, 10))
    ax.grid(
        True,
        linestyle="--",
        linewidth=0.5,
        color="lightgrey",
        alpha=0.7,
    )


def plot_averages_descriptives(role, language, prompt_type):

    filtered = df.filter(
        (pl.col("role") == role)
        & (pl.col("language") == language)
        & (pl.col("type") == prompt_type)
    )

    metrics = [
        ("doc_length", "Text Length (tokens)"),
        ("dependency_distance_mean", "MDD"),
    ]

    aggregated = {
        metric_col: aggregate_metric(filtered, metric_col)
        for metric_col, _ in metrics
    }

    fig, axes = plt.subplots(
        2,
        len(MODELS),
        figsize=(15, 7),
        sharex=True,
        sharey="row",
    )

    token_ylim_map = {
        ("Lithuanian", "constrained"): 350,
        ("Lithuanian", "regular"): 300,
        ("English", "regular"): 350,
        ("English", "constrained"): 350,
    }

    token_ylim = token_ylim_map.get((language, prompt_type))
    dependency_ylim = 2.8

    for col, model in enumerate(MODELS):

        for row, (metric_col, ylabel) in enumerate(metrics):
            plot_metric(
                ax=axes[row, col],
                df=aggregated[metric_col],
                model=model,
                ylabel=ylabel,
            )

        size = model.split("-")[-2].upper()
        axes[0, col].set_title(
            f"Gemma3 {size} IT",
            fontsize=18,
            fontweight="bold",
        )

        if token_ylim is not None:
            axes[0, col].set_ylim(0, token_ylim)

        axes[1, col].set_ylim(0, dependency_ylim)

        axes[0, col].autoscale(enable=False)
        axes[1, col].autoscale(enable=False)

        axes[1, col].set_xlabel("Turn", fontsize=15)

    axes[0, 0].legend(
        title="CEFR",
        fontsize=12,
        title_fontsize=12,
    )

    plt.tight_layout()

    save_path = PATH / f"{language}-{prompt_type}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"Figure saved: {save_path}")


def plot_averages_grey(role, language):

    fig, axes = plt.subplots(
        2,
        len(MODELS),
        figsize=(15, 7),
        sharex=True,
        sharey="row",
    )

    metrics = [
        ("doc_length", "Text Length (tokens)"),
        ("dependency_distance_mean", "MDD"),
    ]

    prompt_styles = {
        "regular": {
            "colors": CEFR_GREY_COLORS,
            "linestyle": "--",
            "alpha": 0.15,
        },
        "constrained": {
            "colors": CEFR_COLORS,
            "linestyle": "-",
            "alpha": 0.20,
        },
    }

    for prompt_type, style in prompt_styles.items():

        filtered = df.filter(
            (pl.col("role") == role)
            & (pl.col("language") == language)
            & (pl.col("type") == prompt_type)
        )

        aggregated = {
            metric_col: aggregate_metric(filtered, metric_col)
            for metric_col, _ in metrics
        }

        for col, model in enumerate(MODELS):

            for row, (metric_col, ylabel) in enumerate(metrics):
                plot_metric(
                    ax=axes[row, col],
                    df=aggregated[metric_col],
                    model=model,
                    ylabel=ylabel,
                    colors=style["colors"],
                    linestyle=style["linestyle"],
                    alpha=style["alpha"],
                    label_suffix=prompt_type,
                )

            size = model.split("-")[-2].upper()
            axes[0, col].set_title(
                f"Gemma3 {size} IT",
                fontsize=18,
                fontweight="bold",
            )

    for col in range(len(MODELS)):
        axes[0, col].set_ylim(0, 350)
        axes[1, col].set_ylim(0, 2.8)

        axes[0, col].autoscale(enable=False)
        axes[1, col].autoscale(enable=False)

        axes[1, col].set_xlabel("Turn", fontsize=15)

    axes[0, 0].legend(
        title="CEFR",
        fontsize=12,
        title_fontsize=12,
    )

    plt.tight_layout()

    save_path = PATH / f"{language}_grey.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"Figure saved: {save_path}")


def main():

    combinations = list(
        product(
            ["assistant"],
            ["English", "Lithuanian"],
            ["regular", "constrained"],
        )
    )

    for role, language, prompt_type in tqdm(
        combinations,
        desc="Generating average plots",
        unit="plot",
    ):
        plot_averages_descriptives(role, language, prompt_type)

    logger.info("Average plots saved.")

    combinations = list(
        product(
            ["assistant"],
            ["English", "Lithuanian"],
        )
    )

    for role, language in tqdm(
        combinations,
        desc="Generating grey comparison plots",
        unit="plot",
    ):
        plot_averages_grey(role, language)

    logger.info("Grey comparison plots saved.")


if __name__ == "__main__":
    main()