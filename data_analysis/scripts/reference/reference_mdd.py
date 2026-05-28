from pathlib import Path

import pandas as pd
import textdescriptives as td

from data_analysis.src.logger import logger

C1_folder = Path("data_analysis/data/output/reference_data/raw_text")
files = list(C1_folder.glob("*.txt"))

pos_files = Path("data_analysis/data/output/reference_data/morph_annotations")


def compute_mdd(text: str, model="lt_core_news_md"):
    metrics = td.extract_metrics(
        text=[text],
        spacy_model=model,
        metrics=["dependency_distance"],
    )
    return pd.DataFrame(metrics)


def read_pos_file(file_path: Path) -> str:
    words = []

    with file_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) < 2:
                continue

            word = parts[0]

            if parts[2] == "PUNCT":
                continue
            words.append(word)

    return " ".join(words)


def get_C1_mdd(files: list):

    all_text = ""

    for file in files:
        text = file.read_text(encoding="utf-8")
        all_text += text + " "

    metrics = td.extract_metrics(
        text=[all_text],
        spacy_model="lt_core_news_md",
        metrics=["dependency_distance"],
    )

    df = pd.DataFrame(metrics)

    mdd = df["dependency_distance_mean"][0]

    return mdd


def main():
    logger.info("Computing the MDD for reference dataset.")

    mdd_c1 = get_C1_mdd(files)

    a1_files = list((pos_files / "a1").rglob("*.pos"))
    b1_files = list((pos_files / "b1").rglob("*.pos"))

    a1_texts = [read_pos_file(f) for f in a1_files]
    b1_texts = [read_pos_file(f) for f in b1_files]

    a1_corpus = "\n".join(a1_texts)
    b1_corpus = "\n".join(b1_texts)

    a1_mdd = compute_mdd(a1_corpus)
    b1_mdd = compute_mdd(b1_corpus)

    df_a1 = pd.DataFrame(a1_mdd)
    df_b1 = pd.DataFrame(b1_mdd)

    mdd_a1 = df_a1["dependency_distance_mean"][0]
    mdd_b1 = df_b1["dependency_distance_mean"][0]

    df_results = pd.DataFrame(
        {
            "A1": [mdd_a1],
            "B1": [mdd_b1],
            "C1": [mdd_c1],
        }
    )

    save_path = Path(
        "data_analysis/data/output/reference_data/sanity_check/reference_mdd.csv"
    )

    df_results.to_csv(save_path, index=False)

    logger.info(f"MDD table saved to {save_path}")


if __name__ == "__main__":
    main()
