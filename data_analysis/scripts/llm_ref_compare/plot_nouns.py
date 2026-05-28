from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from settings import MODEL_COLORS, MODELS, logger

Y_LABELS = {
    "dative_%": "Dative nouns (%)",
    "instrumental_%": "Instrumental nouns (%)",
    "mean_sentence_length": "Average sentence length",
}

YLIMS = {
    "active_present_participle_%": (0, 4),
    "mean_sentence_length": (2, 15),
    "dative_%": (0, 7),
}


def plot_participles_facets(df: pd.DataFrame, save_dir: Path):

    cefr_order = ["A1", "B1", "C1"]

    df = df.copy()
    df["cefr"] = pd.Categorical(df["cefr"], categories=cefr_order, ordered=True)

    reference = df[df["model"] == "reference"]
    llm = df[df["model"] != "reference"]

    metrics = ["dative_%", "instrumental_%", "mean_sentence_length"]

    combos = llm.groupby(["language", "type"])

    for (language, prompt_type), group in combos:
        fig, axes = plt.subplots(
            nrows=len(metrics),
            ncols=1,
            figsize=(4, 3 * len(metrics)),
            sharex=True,
        )

        if len(metrics) == 1:
            axes = [axes]

        fig.suptitle(
            f"{language.capitalize()}\n[{prompt_type} student prompt]",
            fontsize=15,
            fontweight="bold",
            y=0.94,
        )

        for ax, metric in zip(axes, metrics):
            for model in MODELS:
                mgroup = group[group["model"] == model]
                if mgroup.empty:
                    continue

                ax.plot(
                    mgroup["cefr"],
                    mgroup[metric],
                    marker="o",
                    linewidth=1,
                    color=MODEL_COLORS.get(model, "gray"),
                    label=model.split("--")[-1],
                )
            if metric in YLIMS:
                ax.set_ylim(*YLIMS[metric])

            ref_group = (
                reference[reference["cefr"].isin(cefr_order)]
                .groupby("cefr", as_index=False)
                .mean(numeric_only=True)
            )

            ax.plot(
                ref_group["cefr"],
                ref_group[metric],
                linestyle="--",
                linewidth=1,
                color="black",
                label="reference",
            )
            ax.set_title(Y_LABELS.get(metric, metric), fontsize=13, loc="center")
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        if language == "english" and prompt_type == "regular":
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.01),
                fontsize=11,
                ncol=2,
                columnspacing=1.5,
                handletextpad=0.5,
            )

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for title

        save_path = save_dir / f"{language}_{prompt_type}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

        logger.info(f"Saved: {save_path}")


def main():
    file_path = Path(
        "data_analysis/data/output/llm_versus_reference/llm_reference_morph_summary.csv"
    )

    save_dir = Path("data_analysis/plots/ref_llm_compare/nouns")
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(file_path)

    plot_participles_facets(df, save_dir)


if __name__ == "__main__":
    main()
