from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

from data_analysis.src.logger import logger
from settings import LANGUAGE_MAPPING, MODEL_COLORS, MODEL_NAMES, MODELS

path = Path("data_analysis/data/output/mixed_effects/p_values")


def plot_lme_coefficients(file_path: Path):
    """
    Plots linear mixed-effects coefficients as a forest plot.
    """

    metric_labels = {
        "doc_length": "Average Text Length (tokens)",
        "dependency_distance_mean": "Mean Dependency Distance",
    }

    x_axis_positions = {
        "(Intercept)": 0,
        "levelB1": 1,
        "levelC1": 2,
    }

    term_labels = {
        "(Intercept)": "A1",
        "levelB1": "B1",
        "levelC1": "C1",
    }

    metrics = ["doc_length", "dependency_distance_mean"]
    model_offsets = dict(zip(MODELS, [-0.15, 0.0, 0.15]))
    model_display_names = dict(zip(MODELS, MODEL_NAMES))

    # Preparation
    file_parts = file_path.stem.split("_")
    lang_code = file_parts[2]
    prompt_type = file_parts[3]
    language = LANGUAGE_MAPPING.get(lang_code)
    figure_title = f"{language} [{prompt_type} student prompt]"

    # Load the data
    df = pd.read_csv(file_path)

    #df = df[df["term"] != "(Intercept)"].copy()  # Remove the intercept
    df["ci"] = 1.96 * df["std_error"]  # Add CIs for error bars

    df["is_significant"] = (  # Checks if the finding is significant
        df["significance"].notna() & (df["significance"] != "")
    )

    # Create the figure
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(6, 4),
        sharex=True,
        sharey=False,
    )

    fig.suptitle(
        figure_title,
        fontsize=12,
        fontweight="bold",
    )

    # Plot each metric
    for col_idx, metric in enumerate(metrics):
        ax = axes[col_idx]
        metric_df = df[df["metric"] == metric]

        legend_handles = {}

        for model in MODELS:
            model_df = metric_df[metric_df["model"] == model]
            color = MODEL_COLORS[model]
            x_offset = model_offsets[model]

            for _, row in model_df.iterrows():
                term = row["term"]

                if term not in x_axis_positions:
                    continue

                x = x_axis_positions[term] + x_offset
                y = row["estimate"]
                ci = row["ci"]
                significant = row["is_significant"]

                ax.errorbar(
                    x=x,
                    y=y,
                    yerr=ci,
                    fmt="o",
                    color=color,
                    linewidth=1.5,
                    capsize=3,
                    markersize=7,
                    fillstyle="full" if significant else "none",
                    markeredgewidth=1,
                )

            # Add model once to legend
            display_name = model_display_names[model]

            if display_name not in legend_handles:
                legend_handles[display_name] = plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color=color,
                    linestyle="None",
                    markersize=7,
                    label=display_name,
                )

        ax.axhline(
            y=0,
            color="black",
            linewidth=0.8,
            linestyle="--",
        )

        ax.set_xticks(list(x_axis_positions.values()))
        ax.set_xticklabels([term_labels[t] for t in x_axis_positions])

        ax.set_title(
            metric_labels[metric],
            fontsize=10,
        )
        
        if metric == "doc_length":
            ax.set_ylim(0, 230)
        else:
            ax.set_ylim(0, 1.8)

        ax.set_ylabel("Estimate")

        ax.grid(
            True,
            linestyle="--",
            alpha=0.5,
        )

        if metric == "doc_length":
            ax.yaxis.set_major_locator(MultipleLocator(20))
        else:
            ax.yaxis.set_major_locator(MultipleLocator(0.1))

        # Legend adjustment
        if col_idx == 0 and language == "English" and prompt_type == "regular":
            ax.legend(
                handles=list(legend_handles.values()),
                title="Model",
                fontsize=8,
                title_fontsize=8,
                loc="upper left",
            )

    plt.tight_layout()

    save_path = Path(
        f"data_analysis/plots/mixed_effects/estimates/"
        f"{language}_{prompt_type}_coeff.png"
    )

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    logger.info(f"Plot saved at {save_path}.")

    # plt.show()


def main():
    logger.info("Plotting of findings from linear mixed-effects analysis has started.")

    for file in path.iterdir():
        file_dir = path / file.name
        plot_lme_coefficients(file_dir)

    logger.info("Done!")


if __name__ == "__main__":
    main()
