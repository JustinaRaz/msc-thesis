from pathlib import Path
import pandas as pd

from data_analysis.src.logger import logger


def compute_text_stats(folder_path: Path) -> dict:
    total_tokens = 0
    total_word_chars = 0
    sentence_lengths = []
    tokens_list = []

    for file in folder_path.glob("*.pos"):
        current_sentence = []

        with open(file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")

                if len(parts) < 3:
                    if current_sentence:
                        sentence_lengths.append(len(current_sentence))
                        current_sentence = []
                    continue

                upos = parts[2]

                if upos == "PUNCT":
                    continue

                token = parts[0].lower()
                tokens_list.append(token)
                total_tokens += 1
                total_word_chars += len(token)
                current_sentence.append(token)

        if current_sentence:
            sentence_lengths.append(len(current_sentence))

    ttr = round(len(set(tokens_list)) / total_tokens, 3)
    avg_sentence_length = round(sum(sentence_lengths) / len(sentence_lengths), 3)
    avg_word_length = round(total_word_chars / total_tokens, 3)

    return {
        "avg_sentence_length": avg_sentence_length,
        "avg_word_length": avg_word_length,
        "ttr": ttr,
    }


def main():

    input_path = Path("data_analysis/data/output/reference_data/morph_annotations")
    output_path = Path("data_analysis/data/output/reference_data/sanity_check/surface_level_stats.csv")

    rows = []
    for level_dir in sorted(input_path.iterdir()):
        if level_dir.is_dir():
            row = compute_text_stats(level_dir)
            row["level"] = level_dir.name.upper()
            rows.append(row)

    df = pd.DataFrame(rows).set_index("level")
    df.to_csv(output_path)

    logger.info(f"Surface level structural overview has been saved to {output_path}.")

if __name__ == "__main__":
    main()