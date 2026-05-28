from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from settings import logger

df = pl.read_parquet(
    "data_analysis/data/output/metrics/refinement_data_metrics.parquet"
)

MODEL = "google--gemma-3-12b-it"

CEFR_LEVELS = [
    "B1",
    "C1",
]

CEFR_COLORS = {
    "B1": "gold",
    "C1": "darkred",
}


def aggregate_tokens(df):

    return (
        df.group_by(["cefr", "turn"])
        .agg(
            [
                pl.col("doc_length").mean().alias("mean"),
                pl.col("doc_length").std().alias("std"),
                pl.count().alias("n"),
            ]
        )
        .with_columns((1.96 * pl.col("std") / pl.col("n").sqrt()).alias("ci"))
        .sort(["cefr", "turn"])
        .to_pandas()
    )


def aggregate_dependency(df):

    return (
        df.group_by(["cefr", "turn"])
        .agg(
            [
                pl.col("dependency_distance_mean").mean().alias("mean"),
                pl.col("dependency_distance_mean").std().alias("std"),
                pl.count().alias("n"),
            ]
        )
        .with_columns((1.96 * pl.col("std") / pl.col("n").sqrt()).alias("ci"))
        .sort(["cefr", "turn"])
        .to_pandas()
    )


def plot_metric(
    ax,
    metric_df,
    ylabel,
):

    for cefr in CEFR_LEVELS:
        subset = metric_df[metric_df["cefr"] == cefr]

        x = subset["turn"]
        y = subset["mean"]
        ci = subset["ci"]

        ax.plot(
            x,
            y,
            label=cefr,
            color=CEFR_COLORS[cefr],
            linewidth=2,
        )

        ax.fill_between(
            x,
            y - ci,
            y + ci,
            alpha=0.2,
            color=CEFR_COLORS[cefr],
        )

    ax.set_ylabel(ylabel, fontsize=14)

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.5,
        color="gray",
        alpha=0.3,
    )


def plot_text_descriptives(
    language,
):

    filtered = df.filter(pl.col("language") == language)

    tokens = aggregate_tokens(filtered)

    dependency = aggregate_dependency(filtered)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 8),
        sharex=True,
    )

    plot_metric(axes[0], tokens, ylabel="Text Length (tokens)")

    plot_metric(axes[1], dependency, ylabel="MDD")

    axes[0].set_title("Gemma3 12B IT", fontweight="bold", fontsize=16)

    axes[0].legend(title="CEFR", fontsize=12)

    axes[1].set_xlabel("Turn", fontsize=14)

    axes[1].set_ylim(0, 2.8)

    plt.tight_layout()

    save_path = Path(f"data_analysis/plots/refinement/{language}_averages.png")

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    logger.info(f"Averages saved: {save_path}")


def plot_text_descriptive_densities(
    language,
):

    filtered = df.filter(pl.col("language") == language)

    metrics = [
        ("doc_length", "Text Length (tokens)"),
        (
            "dependency_distance_mean",
            "MDD",
        ),
    ]

    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(8, 8),
    )

    for row, (metric_col, xlabel) in enumerate(metrics):
        ax = axes[row]

        for cefr in CEFR_LEVELS:
            cefr_vals = filtered.filter(pl.col("cefr") == cefr)[metric_col].to_list()

            sns.kdeplot(
                cefr_vals,
                ax=ax,
                label=cefr,
                color=CEFR_COLORS[cefr],
                fill=False,
                linewidth=2,
            )

        ax.set_ylabel("Density", fontsize=14)

        axes[0].set_title("Gemma3 12B IT", fontweight="bold", fontsize=16)

        ax.set_xlabel(xlabel, fontsize=14)

        ax.grid(
            True,
            which="major",
            linestyle="--",
            linewidth=0.5,
            color="gray",
            alpha=0.3,
        )

        if metric_col == "dependency_distance_mean":
            ax.set_xlim(0, 7)

    plt.tight_layout()

    save_path = Path(f"data_analysis/plots/refinement/{language}_densities.png")

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )
    logger.info(f"Densities saved: {save_path}")


if __name__ == "__main__":
    plot_text_descriptives(language="Lithuanian")

    plot_text_descriptive_densities(language="Lithuanian")
