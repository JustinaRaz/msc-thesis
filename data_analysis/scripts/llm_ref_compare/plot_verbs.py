from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from settings import MODEL_COLORS, MODELS, logger

Y_LABELS = {
    "finite_verb_%": "Finite verbs (%)",
    "participle_%": "Participles (%)",
    "adverbial_participle_%": "Adverbial participles (%)",
    "half_participle_%": "Half participles (%)",
    "active_present_participle_%": "Active present participles (%)",
}

YLIMS = {
    "finite_verb_%":                (45, 90),
    "participle_%":                 (0, 25),
    "adverbial_participle_%":       (0, 7),
    "half_participle_%":            (0, 2.5),
}

def plot_participles_facets(df: pd.DataFrame, save_dir: Path):

    cefr_order = ["A1", "B1", "C1"]

    df = df.copy()
    df["cefr"] = pd.Categorical(df["cefr"], categories=cefr_order, ordered=True)

    reference = df[df["model"] == "reference"]
    llm = df[df["model"] != "reference"]

    metrics = [
        "finite_verb_%",
        "participle_%",
        "adverbial_participle_%",
        "half_participle_%",
        "active_present_participle_%",
    ]

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
            y=0.95,
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

            ref_group = reference[
                reference["cefr"].isin(cefr_order)
            ].groupby("cefr", as_index=False).mean(numeric_only=True)

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
                fontsize=12,
                ncol=2,
                columnspacing=1.5,
                handletextpad=0.5,
            )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        save_path = save_dir / f"{language}_{prompt_type}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

        logger.info(f"Saved: {save_path}")


def main():
    file_path = Path(
        "data_analysis/data/output/llm_versus_reference/llm_reference_morph_summary.csv"
    )

    save_dir = Path(
        "data_analysis/plots/ref_llm_compare/verbs"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(file_path)

    plot_participles_facets(df, save_dir)


if __name__ == "__main__":
    main()