from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from settings import CEFR_COLORS, CEFR_LEVELS, MODEL_NAMES, MODELS

DATA_PATH = Path("data_analysis/data/output/metrics/llm_data_metrics.parquet")
SAVE_DIR = Path("data_analysis/plots/distance")

METRICS = {
    "dependency_distance_mean": {
        "distance_col": "dependency_distance_mean_dist",
        "ylabel": "MDD Distance",
        "folder": "mdd",
        "filename": "MDD_distance_plot",
        "ylim": 1,
        "special_ylim": {
            ("google--gemma-3-4b-it", "Lithuanian", "regular"): 2,
        },
    },
    "doc_length": {
        "distance_col": "doc_length_dist",
        "ylabel": "Text Length Distance",
        "folder": "text_length",
        "filename": "text_length_distance_plot",
        "ylim": 150,
        "special_ylim": {},
    },
}


def load_distance_data():
    df = pl.read_parquet(DATA_PATH).to_pandas()

    df_tutor = df[df["role"] == "assistant"].copy()
    df_student = df[df["role"] == "user"].copy()

    df_merged = df_tutor.merge(
        df_student,
        on=["dialogue_id", "turn"],
        suffixes=("_tutor", "_student"),
    )

    for metric in METRICS:
        df_merged[f"{metric}_dist"] = (
            df_merged[f"{metric}_tutor"] - df_merged[f"{metric}_student"]
        ).abs()

    distance_cols = [config["distance_col"] for config in METRICS.values()]

    distance_df = df_merged[
        [
            "dialogue_id",
            "turn",
            "model_tutor",
            "language_tutor",
            "type_tutor",
            "cefr_tutor",
        ]
        + distance_cols
    ]

    distance_df = distance_df.rename(
        columns={
            "model_tutor": "model",
            "language_tutor": "language",
            "type_tutor": "type",
            "cefr_tutor": "cefr",
        }
    )

    return distance_df


def compute_confidence_intervals(df, value_col):
    grouped = df.groupby(
        ["language", "type", "model", "cefr", "turn"],
        as_index=False,
    ).agg(
        mean=(value_col, "mean"),
        std=(value_col, "std"),
        count=(value_col, "count"),
    )

    grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"])
    grouped["ci_low"] = grouped["mean"] - 1.96 * grouped["sem"]
    grouped["ci_high"] = grouped["mean"] + 1.96 * grouped["sem"]

    return grouped


def plot_distance_facets(df, metric, config):
    df = df.copy()

    df["cefr"] = pd.Categorical(
        df["cefr"],
        categories=CEFR_LEVELS,
        ordered=True,
    )

    summary_df = compute_confidence_intervals(
        df=df,
        value_col=config["distance_col"],
    )

    model_name_map = dict(zip(MODELS, MODEL_NAMES))

    for (language, prompt_type), group in summary_df.groupby(["language", "type"]):
        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(MODELS),
            figsize=(6 * len(MODELS), 6),
            sharex=False,
            sharey=False,
        )

        axes = np.atleast_1d(axes)

        for i, model in enumerate(MODELS):
            ax = axes[i]
            mgroup = group[group["model"] == model]

            for cefr in CEFR_LEVELS:
                sub = mgroup[mgroup["cefr"] == cefr]

                if sub.empty:
                    continue

                ax.fill_between(
                    sub["turn"],
                    sub["ci_low"],
                    sub["ci_high"],
                    color=CEFR_COLORS[cefr],
                    alpha=0.2,
                )

                ax.plot(
                    sub["turn"],
                    sub["mean"],
                    marker="o",
                    linewidth=1.5,
                    color=CEFR_COLORS[cefr],
                    label=cefr,
                )

            ax.set_title(
                model_name_map.get(model, model),
                fontsize=20,
                fontweight="bold",
            )

            if i == 0:
                ax.set_ylabel(config["ylabel"], fontsize=16)

            ax.set_xlabel("Turn", fontsize=16)

            ax.grid(
                axis="y",
                linestyle="--",
                alpha=0.5,
            )

            ylim = config["special_ylim"].get(
                (model, language, prompt_type),
                config["ylim"],
            )
            ax.set_ylim(0, ylim)

            ax.tick_params(
                axis="both",
                labelsize=11,
            )

        fig.suptitle(
            f"{language} [{prompt_type} student prompt]",
            fontsize=24,
            fontweight="bold",
        )

        handles = [
            plt.Line2D(
                [0],
                [0],
                color=CEFR_COLORS[cefr],
                marker="o",
                linestyle="-",
            )
            for cefr in CEFR_LEVELS
        ]

        if language == "English" and prompt_type == "regular":
            fig.legend(
                handles,
                CEFR_LEVELS,
                title="CEFR",
                loc="upper right",
                fontsize=13,
                title_fontsize=15,
            )

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        save_path = (
            SAVE_DIR
            / config["folder"]
            / f"{language}_{prompt_type}_{config['filename']}.png"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()


def main():
    distance_df = load_distance_data()

    for metric, config in METRICS.items():
        plot_distance_facets(
            df=distance_df,
            metric=metric,
            config=config,
        )


if __name__ == "__main__":
    main()
