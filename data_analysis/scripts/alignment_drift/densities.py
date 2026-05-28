from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from data_analysis.src.logger import logger
from settings import CEFR_COLORS, CEFR_GREY_COLORS, CEFR_LEVELS, MODELS
from tqdm import tqdm

df = pl.read_parquet("data_analysis/data/metrics/metrics.parquet")

PATH = Path("data_analysis/plots/alignment_drift/densities")

def plot_individual_densities(role, language, prompt_type):

    filtered = df.filter(
        (pl.col("role") == role)
        & (pl.col("language") == language)
        & (pl.col("type") == prompt_type)
    )

    metrics = [
        ("doc_length", "Text Length"),
        ("dependency_distance_mean", "MDD"),
    ]

    ylim_map = {
        ("Lithuanian", "regular"): (0.03, 1.7),
        ("English", "regular"): (0.03, 1.7),
        ("Lithuanian", "constrained"): (0.05, 1.9),
        ("English", "constrained"): (0.05, 1.9),
    }

    fig, axes = plt.subplots(
        len(metrics),
        len(MODELS),
        figsize=(15, 7),
        sharey="row",
    )

    if len(metrics) == 1:
        axes = [axes]

    for row, (metric_col, xlabel) in enumerate(metrics):
        for col, model in enumerate(MODELS):
            ax = axes[row, col]
            model_df = filtered.filter(pl.col("model") == model)

            for cefr in CEFR_LEVELS:
                cefr_vals = (
                    model_df.filter(pl.col("cefr") == cefr)
                    .get_column(metric_col)
                    .to_list()
                )

                if not cefr_vals:
                    continue

                sns.kdeplot(
                    cefr_vals,
                    ax=ax,
                    label=cefr,
                    color=CEFR_COLORS[cefr],
                    fill=False,
                    linewidth=2,
                )

            if row == 0:
                size = model.split("-")[-2].upper()
                ax.set_title(
                    f"Gemma3 {size} IT",
                    fontsize=18,
                    fontweight="bold",
                )

            ax.set_xlabel(xlabel, fontsize=15)

            if col == 0:
                ax.set_ylabel("Density", fontsize=15)

            ax.grid(
                True,
                which="major",
                linestyle="--",
                linewidth=0.5,
                color="gray",
                alpha=0.3,
            )

            ax.tick_params(axis="both", labelsize=10)

            if metric_col == "dependency_distance_mean":
                ax.set_xlim(0, 7)

    ylim_doc, ylim_dep = ylim_map.get((language, prompt_type), (None, None))

    if ylim_doc is not None:
        for ax in axes[0, :]:
            ax.set_ylim(0, ylim_doc)

    if ylim_dep is not None:
        for ax in axes[1, :]:
            ax.set_ylim(0, ylim_dep)

    axes[0, -1].legend(
        title="CEFR",
        fontsize=12,
        title_fontsize=12,
        loc="upper right",
    )

    plt.tight_layout()

    save_path = PATH / f"{language}-{prompt_type}-densities.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"Figure saved: {save_path}")


def plot_combined_densities_combined_grey(role, language):

    metrics = [
        ("doc_length", "Text Length"),
        ("dependency_distance_mean", "MDD"),
    ]

    prompt_styles = {
        "regular": {
            "colors": CEFR_GREY_COLORS,
            "fill": True,
            "alpha": 0.2,
            "linewidth": 0.75,
        },
        "constrained": {
            "colors": CEFR_COLORS,
            "fill": False,
            "alpha": 1.0,
            "linewidth": 2,
        },
    }

    fig, axes = plt.subplots(
        len(metrics),
        len(MODELS),
        figsize=(15, 7),
        sharey="row",
    )

    if len(metrics) == 1:
        axes = [axes]

    for prompt_type, style in prompt_styles.items():
        filtered = df.filter(
            (pl.col("role") == role)
            & (pl.col("language") == language)
            & (pl.col("type") == prompt_type)
        )

        for row, (metric_col, xlabel) in enumerate(metrics):
            for col, model in enumerate(MODELS):
                ax = axes[row, col]
                model_df = filtered.filter(pl.col("model") == model)

                for cefr in CEFR_LEVELS:
                    cefr_vals = (
                        model_df.filter(pl.col("cefr") == cefr)
                        .get_column(metric_col)
                        .to_list()
                    )

                    if not cefr_vals:
                        continue

                    sns.kdeplot(
                        cefr_vals,
                        ax=ax,
                        label=f"{cefr} ({prompt_type})",
                        color=style["colors"][cefr],
                        fill=style["fill"],
                        alpha=style["alpha"],
                        linewidth=style["linewidth"],
                    )

                if row == 0:
                    size = model.split("-")[-2].upper()
                    ax.set_title(
                        f"Gemma3 {size} IT",
                        fontsize=18,
                        fontweight="bold",
                    )

                ax.set_xlabel(xlabel, fontsize=15)

                if col == 0:
                    ax.set_ylabel("Density", fontsize=15)

                ax.grid(
                    True,
                    which="major",
                    linestyle="--",
                    linewidth=0.5,
                    color="gray",
                    alpha=0.3,
                )

                ax.tick_params(axis="both", labelsize=10)

                if metric_col == "dependency_distance_mean":
                    ax.set_xlim(0, 7)

    for ax in axes[0, :]:
        ax.set_ylim(0, 0.05)

    for ax in axes[1, :]:
        ax.set_ylim(0, 1.9)

    axes[0, -1].legend(
        fontsize=14,
        title_fontsize=14,
        loc="upper right",
    )

    plt.tight_layout()

    save_path = PATH / f"{language}-combined-densities_grey.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"Figure saved: {save_path}")


def main():

    individual_combinations = list(
        product(
            ["assistant"],
            ["English", "Lithuanian"],
            ["regular", "constrained"],
        )
    )

    for role, language, prompt_type in tqdm(
        individual_combinations,
        desc="Generating plots",
        unit="plot",
    ):
        plot_individual_densities(role, language, prompt_type)

    logger.info("Individual density plots saved.")

    combinations = list(
        product(
            ["assistant"],
            ["English", "Lithuanian"],
        )
    )

    for role, language in tqdm(
        combinations,
        desc="Generating plots",
        unit="plot",
    ):
        plot_combined_densities_combined_grey(role, language)

    logger.info("Density comparison plots saved.")


if __name__ == "__main__":
    main()
