from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from data_analysis.src.logger import logger
from settings import MODEL_COLORS, MODEL_NAMES, MODELS

df = pl.read_csv("data_analysis/data/metrics/cleaning_report.csv")


def plot_cleaning_report(df):

    metrics = [
        "repetitive_words_removed",
        "invalid_sentences_removed",
        "long_words_removed",
        "invalid_char_words_removed",
    ]
    titles = [
        "Repetitive Words",
        "English Sentences",
        "Long Words",
        "Non-Lithuanian Words",
    ]

    agg = df.group_by(["model", "prompt_type", "language"]).agg(
        [pl.col(m).sum() for m in metrics]
    )

    conditions = [
        ("English", "regular"),
        ("Lithuanian", "regular"),
        ("English", "constrained"),
        ("Lithuanian", "constrained"),
    ]

    condition_labels = [f"{lang}\n[{pt}]" for lang, pt in conditions]

    bar_width = 0.25 / (len(MODELS) / 3)
    x = np.arange(len(conditions))

    for metric, title in zip(metrics, titles):
        fig, ax = plt.subplots(figsize=(10, 5))

        for k, model in enumerate(MODELS):
            model_data = agg.filter(pl.col("model") == model)
            values = []

            for lang, pt in conditions:
                row = model_data.filter(
                    (pl.col("language") == lang) & (pl.col("prompt_type") == pt)
                )
                values.append(row[metric][0] if row.height > 0 else 0)

            bar_positions = x + k * bar_width
            bars = ax.bar(
                bar_positions,
                values,
                width=bar_width,
                color=MODEL_COLORS[model],
                label=MODEL_NAMES[k],
                alpha=0.5,
                zorder=2,
            )
            ax.bar_label(bars, fmt="%d", padding=3, fontsize=11)

        ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.15)
        ax.set_xticks(x + bar_width * (len(MODELS) - 1) / 2)
        ax.set_xticklabels(condition_labels, rotation=0, fontsize=15)
        ax.set_ylabel("Total removed", fontsize=15)
        ax.set_title(title, fontsize=22, fontweight="bold")
        if metric == "invalid_sentences_removed":
            ax.legend(title="Model")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()

        save_path = Path(f"data_analysis/plots/cleaning_report/{metric}.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Figure saved:{save_path}")


def main():

    plot_cleaning_report(df)


if __name__ == "__main__":
    main()
