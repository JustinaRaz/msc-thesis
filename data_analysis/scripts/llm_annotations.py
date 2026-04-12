import pandas as pd
from pathlib import Path
import stanza
from data_analysis.src.logger import logger


def get_stanza():
    """
    Initialize Stanza (download only once).
    """
    logger.info("Initializing Stanza pipeline.")

    stanza.download("lt")  # will skip if already downloaded

    processor = stanza.Pipeline(
        lang="lt",
        processors="tokenize,pos,lemma",
        tokenize_pretokenized=False
    )

    return processor


def annotate_and_save(df, output_dir):

    processor = get_stanza()

    # Group by relevant columns
    grouped = df.groupby(["model", "language", "type", "cefr"])

    for (model, language, type_, cefr), group in grouped:

        logger.info(f"Processing: {model} | {language} | {type_} | {cefr}")

        # ⚠️ Ensure correct order
        group = group.sort_values("turn")

        # Combine all content into one text
        text = "\n".join(group["content"].astype(str).tolist())

        if not text.strip():
            logger.warning("Empty text, skipping...")
            continue

        # Run Stanza
        doc = processor(text)

        # Prepare output directory
        model_dir = Path(output_dir) / model
        model_dir.mkdir(parents=True, exist_ok=True)

        # Clean filename (safe)
        language_clean = language.lower()
        type_clean = type_.lower()

        file_name = f"{language_clean}_{type_clean}_{cefr}.pos"
        file_path = model_dir / file_name

        # Write POS annotations
        with open(file_path, "w", encoding="utf-8") as f:
            for sentence in doc.sentences:
                for word in sentence.words:
                    f.write(f"{word.text}\t{word.upos}\t{word.lemma}\n")
                f.write("\n")

        logger.info(f"Saved: {file_path}")


def main():
    input_path = Path("data_analysis/data/llm_text/clean_dataset.parquet")
    output_dir = Path("data_analysis/data/morphological_annotations/llm_simulated_data")

    # 👉 Read with Polars, convert to Pandas
    import polars as pl
    df = pl.read_parquet(input_path).to_pandas()

    # Filter role
    df = df[df["role"] == "assistant"]

    annotate_and_save(df, output_dir)


if __name__ == "__main__":
    main()