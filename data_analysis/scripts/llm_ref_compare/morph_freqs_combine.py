from pathlib import Path

import pandas as pd
from settings import logger


def load_reference_data():

    base_path = Path("data_analysis/data/output/reference_data/sanity_check")

    morph_df = pd.read_csv(base_path / "morph_freqs_final.csv")

    surface_df = pd.read_csv(base_path / "surface_level_stats.csv")

    mdd_df = pd.read_csv(base_path / "reference_mdd.csv")

    mdd_long = mdd_df.iloc[0].reset_index()
    mdd_long.columns = [
        "cefr",
        "dependency_distance_mean",
    ]

    morph_df = morph_df.rename(columns={"level": "cefr"})

    surface_df = surface_df.rename(
        columns={
            "level": "cefr",
            "avg_sentence_length": "mean_sentence_length",
            "ttr": "ttr_surface",
        }
    )
    reference_df = morph_df.merge(surface_df, on="cefr", how="left").merge(
        mdd_long, on="cefr", how="left"
    )

    reference_df["model"] = "reference"
    reference_df["language"] = "lithuanian"
    reference_df["type"] = "reference"

    first_cols = [
        "model",
        "language",
        "type",
        "cefr",
    ]

    remaining_cols = [col for col in reference_df.columns if col not in first_cols]

    reference_df = reference_df[first_cols + remaining_cols]

    return reference_df


def combine_with_llm_data():
    """
    Combine reference dataset with morph_summary.csv
    using shared columns only.
    """

    llm_path = Path("data_analysis/data/output/llm/morph_summaries/morph_summary.csv")

    llm_df = pd.read_csv(llm_path)
    llm_df = llm_df.dropna(how="all")

    reference_df = load_reference_data()

    # Keep only shared columns
    shared_columns = [col for col in llm_df.columns if col in reference_df.columns]

    combined_df = pd.concat(
        [
            llm_df[shared_columns],
            reference_df[shared_columns],
        ],
        ignore_index=True,
    )

    save_path = Path("data_analysis/data/output/llm_versus_reference/llm_reference_morph_summary.csv")

    combined_df.to_csv(save_path, index=False)

    logger.info(f"Saved to: {save_path}")


if __name__ == "__main__":
    combine_with_llm_data()
