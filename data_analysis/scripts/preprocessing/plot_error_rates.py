from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from data_analysis.src.logger import logger
from data_analysis.src.metrics import MetricExtractor
from matplotlib.ticker import MultipleLocator
from settings import MODEL_COLORS, MODEL_NAMES, MODELS

df_path = "data_analysis/data/output/llm/llm_text/clean_dataset.parquet"


def plot_error_rates(df):

    languages = sorted(df["language"].unique())
    types = sorted(df["type"].unique())
    cefr_order = ["A1", "B1", "C1"]
    model_label = dict(zip(MODELS, MODEL_NAMES))

    for i, language in enumerate(languages):
        for j, prompt_type in enumerate(types):
            fig, ax = plt.subplots(figsize=(6, 5))

            subset = df[(df["language"] == language) & (df["type"] == prompt_type)]

            for model in MODELS:
                model_subset = subset[subset["model"] == model]

                if model_subset.empty:
                    continue

                x_vals = []
                y_vals = []
                ci_vals = []

                for cefr in cefr_order:
                    row = model_subset[model_subset["cefr"] == cefr]

                    if len(row) == 0:
                        continue

                    x_vals.append(cefr)
                    y_vals.append(row["mean_error"].values[0])
                    ci_vals.append(row["ci_95"].values[0])

                if not x_vals:
                    continue

                ax.plot(
                    x_vals,
                    y_vals,
                    marker="o",
                    linewidth=2,
                    color=MODEL_COLORS[model],
                    label=model_label[model],
                )

                ax.fill_between(
                    x_vals,
                    [y - ci for y, ci in zip(y_vals, ci_vals)],
                    [y + ci for y, ci in zip(y_vals, ci_vals)],
                    color=MODEL_COLORS[model],
                    alpha=0.2,
                )

            ax.set_title(f"{language.capitalize()} [{prompt_type} student prompt]")
            ax.set_ylabel("Proportion of Incorrect Words")
            ax.set_xlabel("CEFR")
            ax.tick_params(axis="x", labelbottom=True)
            ax.tick_params(axis="y", labelleft=True)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.yaxis.set_major_locator(MultipleLocator(0.01))
            ax.set_ylim(0, 0.12)

            if language == "English" and prompt_type == "regular":
                ax.legend(title="Model")

            plt.tight_layout()

            save_path = Path(
                f"data_analysis/plots/error_rate/{language}-{prompt_type}.png"
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Figure saved: {save_path}")


def main():

    extractor = MetricExtractor(data_file=df_path)

    df = extractor.compute_sentence_error_rates()

    logger.info("Computing average error rate per dialogue.")

    dialogue_error = df.group_by(
        ["dialogue_id", "model", "language", "cefr", "type"]
    ).agg(pl.col("sentence_error_rate").mean().alias("dialogue_error_rate"))

    condition_error = (
        dialogue_error.group_by(["model", "language", "cefr", "type"])
        .agg(
            [
                pl.col("dialogue_error_rate").mean().alias("mean_error"),
                pl.col("dialogue_error_rate").std().alias("std_error"),
                pl.len().alias("n_dialogues"),
            ]
        )
        .with_columns(
            (1.96 * pl.col("std_error") / pl.col("n_dialogues").sqrt()).alias("ci_95")
        )
        .sort(["language", "type", "model", "cefr"])
    )

    df_plot = condition_error.to_pandas()

    plot_error_rates(df_plot)


if __name__ == "__main__":
    main()
