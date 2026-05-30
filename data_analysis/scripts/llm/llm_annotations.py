from pathlib import Path

import polars as pl
import stanza
from data_analysis.src.logger import logger


def get_stanza():
    """
    Initialize Stanza (download only once).
    """
    logger.info("Initializing Stanza pipeline.")

    stanza.download("lt")  # will skip if already downloaded

    processor = stanza.Pipeline(
        lang="lt", processors="tokenize,pos,lemma", tokenize_pretokenized=False
    )

    return processor


def annotate_and_save(df, output_dir):
    processor = get_stanza()

    grouped = df.groupby(["model", "language", "type", "cefr"])

    for (model, language, type_, cefr), group in grouped:
        logger.info(f"Processing: {model} | {language} | {type_} | {cefr}")

        group = group.sort_values("turn")
        text = "\n".join(group["content"].astype(str).tolist())

        if not text.strip():
            continue

        doc = processor(text)

        model_dir = Path(output_dir) / model
        model_dir.mkdir(parents=True, exist_ok=True)

        file_path = model_dir / f"{language.lower()}_{type_.lower()}_{cefr}.pos"

        with open(file_path, "w", encoding="utf-8") as f:
            for sentence in doc.sentences:
                for word in sentence.words:
                    xpos = word.xpos or "_"
                    feats = word.feats or "_"

                    f.write(
                        f"{word.text}\t{word.lemma}\t{word.upos}\t{xpos}\t{feats}\n"
                    )
                f.write("\n")

        logger.info(f"Saved: {file_path}")


def main():
    input_path = Path("data_analysis/data/output/llm/llm_text/clean_dataset.parquet")
    output_dir = Path("data_analysis/data/output/llm/morph_annotations")

    df = pl.read_parquet(input_path).to_pandas()

    df = df[df["role"] == "assistant"]

    annotate_and_save(df, output_dir)


if __name__ == "__main__":
    main()
