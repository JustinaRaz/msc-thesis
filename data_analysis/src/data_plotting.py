import matplotlib.pyplot as plt
import polars as pl
from pathlib import Path
from matplotlib.ticker import MultipleLocator
import numpy as np


class Plotter:
    def __init__(self):

        self.df = pl.read_parquet("data_analysis/data/metrics.parquet")

        self.models = [
            "google--gemma-3-4b-it",
            "google--gemma-3-12b-it",
            "google--gemma-3-27b-it",
        ]

        self.cefr_order = [
            "A1", 
            "B1", 
            "C1"]

        self.cefr_colors = {
            "A1": "olivedrab",
            "B1": "gold",
            "C1": "darkred",
        }

        self.model_colors = {
            "google--gemma-3-4b-it": "olivedrab",
            "google--gemma-3-12b-it": "gold",
            "google--gemma-3-27b-it": "darkred",
        }

        self.available_roles = sorted(self.df["role"].unique())
        self.available_languages = sorted(self.df["language"].unique())
        self.available_types = sorted(self.df["type"].unique())

    def _validate(self, role, language, prompt_type):

        if role not in self.available_roles:
            raise ValueError(f"Role must be one of {self.available_roles}")

        if language not in self.available_languages:
            raise ValueError(f"Language must be one of {self.available_languages}")

        if prompt_type not in self.available_types:
            raise ValueError(f"Type must be one of {self.available_types}")

    def _aggregate_tokens(self, df):

        return (
            df.group_by(["model", "cefr", "turn"])
            .agg(
                [
                    pl.col("n_tokens").mean().alias("mean"),
                    pl.col("n_tokens").std().alias("std"),
                    pl.count().alias("n"),
                ]
            )
            .with_columns((1.96 * pl.col("std") / pl.col("n").sqrt()).alias("ci"))
            .sort(["model", "cefr", "turn"])
            .to_pandas()
        )

    def _aggregate_sentence_length(self, df):

        return (
            df.group_by(["model", "cefr", "turn"])
            .agg(
                [
                    pl.col("sentence_length_mean").mean().alias("mean"),
                    pl.col("sentence_length_mean").std().alias("std"),
                    pl.count().alias("n"),
                ]
            )
            .with_columns((1.96 * pl.col("std") / pl.col("n").sqrt()).alias("ci"))
            .sort(["model", "cefr", "turn"])
            .to_pandas()
        )

    def _aggregate_dependency(self, df):

        return (
            df.group_by(["model", "cefr", "turn"])
            .agg(
                [
                    pl.col("dependency_distance_mean").mean().alias("mean"),
                    pl.col("dependency_distance_mean").std().alias("std"),
                    pl.count().alias("n"),
                ]
            )
            .with_columns((1.96 * pl.col("std") / pl.col("n").sqrt()).alias("ci"))
            .sort(["model", "cefr", "turn"])
            .to_pandas()
        )

    def plot_text_descriptives(self, role, language, prompt_type):

        self._validate(role, language, prompt_type)

        filtered = self.df.filter(
            (pl.col("role") == role)
            & (pl.col("language") == language)
            & (pl.col("type") == prompt_type)
        )

        tokens = self._aggregate_tokens(filtered)
        sentence = self._aggregate_sentence_length(filtered)
        dependency = self._aggregate_dependency(filtered)

        fig, axes = plt.subplots(3, len(self.models), figsize=(15, 10), sharex=True)

        for col, model in enumerate(self.models):
            self._plot_metric(
                axes[0, col], tokens, model, ylabel="Average Text Length (tokens)"
            )

            self._plot_metric(
                axes[1, col], sentence, model, ylabel="Average Sentence Length"
            )

            self._plot_metric(
                axes[2, col], dependency, model, ylabel="Mean Dependency Distance"
            )

            size = model.split("-")[-2].upper()
            axes[0, col].set_title(f"Gemma {size} IT")

        axes[0, 0].legend(title="CEFR")

        plt.tight_layout()
        
        save_path = Path(f"data_analysis/plots/text_descriptives/{language}-{prompt_type}.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def _plot_metric(self, ax, df, model, ylabel):

        model_df = df[df["model"] == model]

        for cefr in sorted(model_df["cefr"].unique()):
            color = self.cefr_colors.get(cefr, "black")
            subset = model_df[model_df["cefr"] == cefr]

            ax.plot(
                subset["turn"], subset["mean"], color=color, linewidth=2, label=cefr
            )

            ax.fill_between(
                subset["turn"],
                subset["mean"] - subset["ci"],
                subset["mean"] + subset["ci"],
                color=color,
                alpha=0.2,
            )

        ax.set_ylabel(ylabel)
        ax.set_xticks(range(1, 10))
        ax.grid(True, linestyle="--", linewidth=0.5, color="lightgrey", alpha=0.7)

    def plot_error_rates(self, df_plot):

        languages = sorted(df_plot["language"].unique())
        types = sorted(df_plot["type"].unique())
        cefr_order = ["A1", "B1", "C1"]

        fig, axes = plt.subplots(
            len(languages),
            len(types),
            figsize=(6 * len(types), 5 * len(languages)),
            sharey=True
        )

        if len(languages) == 1:
            axes = axes.reshape(1, -1)
        if len(types) == 1:
            axes = axes.reshape(-1, 1)

        for i, language in enumerate(languages):
            for j, prompt_type in enumerate(types):

                ax = axes[i, j]

                subset = df_plot[
                    (df_plot["language"] == language) &
                    (df_plot["type"] == prompt_type)
                ]

                for model in self.models:

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
                        color=self.model_colors[model],
                        label=model
                    )

                    ax.fill_between(
                        x_vals,
                        [y - ci for y, ci in zip(y_vals, ci_vals)],
                        [y + ci for y, ci in zip(y_vals, ci_vals)],
                        color=self.model_colors[model],
                        alpha=0.2
                    )


                ax.set_title(f"{language.capitalize()} [{prompt_type} student prompt]")

                if j == 0:
                    ax.set_ylabel("Total Errors")
                else:
                    ax.set_ylabel("")

                if i == len(languages) - 1:
                    ax.set_xlabel("CEFR")
                else:
                    ax.set_xlabel("")

                ax.tick_params(axis="x", labelbottom=True)
                ax.tick_params(axis="y", labelleft=True)

                ax.grid(True, linestyle="--", alpha=0.5)
                ax.yaxis.set_major_locator(MultipleLocator(0.01))

                if i == 0 and j == 0:
                    ax.legend(title="Model")

        plt.tight_layout()
        save_path = Path("data_analysis/plots/error_rate/error_rate.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_cleaning_report(self, data):

        metrics = [
            "repetitive_words_removed",
            "invalid_sentences_removed",
            "long_words_removed",
            "invalid_char_words_removed",
        ]
        titles = [
            "Repetitive Words",
            "Invalid Sentences",
            "Long Words",
            "Invalid Character Words",
        ]

        # Aggregate: sum per (model, prompt_type, language)
        agg = (
            data.group_by(["model", "prompt_type", "language"])
            .agg([pl.col(m).sum() for m in metrics])
        )

        # Build condition labels e.g. "constrained / English"
        conditions = sorted(
            agg.select(["prompt_type", "language"])
            .unique()
            .rows(),
            key=lambda r: (r[0], r[1])  # sort by prompt_type then language
        )
        condition_labels = [f"{pt} / {lang}" for pt, lang in conditions]

        bar_width = 0.25 / (len(self.models) / 3)  # scale width to number of models
        x = np.arange(len(conditions))

        for metric, title in zip(metrics, titles):

            fig, ax = plt.subplots(figsize=(10, 5))

            for k, model in enumerate(self.models):

                model_data = agg.filter(pl.col("model") == model)
                values = []

                for pt, lang in conditions:
                    row = model_data.filter(
                        (pl.col("prompt_type") == pt) &
                        (pl.col("language") == lang)
                    )
                    values.append(row[metric][0] if row.height > 0 else 0)

                bar_positions = x + k * bar_width
                bars = ax.bar(                          # capture the return value
                    bar_positions, values,
                    width=bar_width,
                    color=self.model_colors[model],
                    label=model,
                    alpha=0.85,
                    zorder=2
                )
                ax.bar_label(bars, fmt="%d", padding=3, fontsize=8)  # label immediately after

            ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.15)  # headroom for labels
            ax.set_xticks(x + bar_width * (len(self.models) - 1) / 2)
            condition_labels = [f"{lang}\n{pt}" for pt, lang in conditions]
            ax.set_xticklabels(condition_labels, rotation=0)
            ax.set_ylabel("Total removed")
            ax.set_title(title, fontsize=13, fontweight="bold")
            if metric == "invalid_sentences_removed":
                ax.legend(title="Model")
            ax.grid(True, axis="y", linestyle="--", alpha=0.5)
            ax.set_ylim(bottom=0)

            plt.tight_layout()

            save_path = Path(f"data_analysis/plots/cleaning_report/{metric}.png")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.show()