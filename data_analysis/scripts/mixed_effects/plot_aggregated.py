from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from settings import MODEL_COLORS, MODEL_NAMING, logger

# datasets to plot:
plot_df = {
    "RQ_1": "data_analysis/data/output/mixed_effects/p_values/results_RQ1_EN_regular_pvalues.csv",
    "RQ_2": "data_analysis/data/output/mixed_effects/p_values/results_RQ2_LT_regular_pvalues.csv",
    "RQ_3_EN": "data_analysis/data/output/mixed_effects/p_values/results_RQ3_EN_constrained_pvalues.csv",
    "RQ_3_LT": "data_analysis/data/output/mixed_effects/p_values/results_RQ3_LT_constrained_pvalues.csv",
}


def plot_aggregated_estimates(
    dataset_path: str | Path,
    output_folder_name: str,
    figsize: tuple = (8, 5),
):

    dataset_path = Path(dataset_path)

    save_dir = (
        Path("data_analysis")
        / "plots"
        / "mixed_effects"
        / "aggregated"
        / output_folder_name
    )

    save_dir.mkdir(parents=True, exist_ok=True)

    metric_titles = {
        "dependency_distance_mean": "Mean Dependency Distance Across CEFR Levels",
        "doc_length": "Text Length Across CEFR Levels",
    }

    metric_ylabels = {
        "dependency_distance_mean": "Estimated values",
        "doc_length": "Estimated values",
    }

    df = pd.read_csv(dataset_path)

    df = df[df["term"].isin(["(Intercept)", "levelB1", "levelC1"])]

    rows = []

    for model in df["model"].dropna().unique():
        for metric in df["metric"].dropna().unique():
            subset = df[(df["model"] == model) & (df["metric"] == metric)]

            if subset.empty:
                continue

            intercept = subset.loc[subset["term"] == "(Intercept)", "estimate"].iloc[0]

            b1 = subset.loc[subset["term"] == "levelB1", "estimate"].iloc[0]

            c1 = subset.loc[subset["term"] == "levelC1", "estimate"].iloc[0]

            rows.extend(
                [
                    {
                        "model": model,
                        "metric": metric,
                        "level": "A1",
                        "value": intercept,
                    },
                    {
                        "model": model,
                        "metric": metric,
                        "level": "B1",
                        "value": intercept + b1,
                    },
                    {
                        "model": model,
                        "metric": metric,
                        "level": "C1",
                        "value": intercept + c1,
                    },
                ]
            )

    plot_df = pd.DataFrame(rows)

    for metric in plot_df["metric"].unique():
        metric_df = plot_df[plot_df["metric"] == metric]

        plt.figure(figsize=figsize)

        for model in metric_df["model"].unique():
            subset = metric_df[metric_df["model"] == model]

            plt.plot(
                subset["level"],
                subset["value"],
                marker="o",
                linewidth=2,
                markersize=8,
                color=MODEL_COLORS.get(model, "black"),
                label=MODEL_NAMING.get(model, model),
            )

        plt.title(metric_titles.get(metric, metric))

        plt.xlabel("CEFR Level")

        plt.ylabel(metric_ylabels.get(metric, "Estimated Value"))

        plt.legend(title="Model")

        plt.grid(alpha=0.3)

        plt.tight_layout()

        plt.savefig(
            save_dir / f"{metric}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    logger.info(f"Plots saved to: {save_dir}")


def main():

    for folder, path in plot_df.items():
        plot_aggregated_estimates(
            dataset_path=path,
            output_folder_name=folder,
        )


if __name__ == "__main__":
    main()
