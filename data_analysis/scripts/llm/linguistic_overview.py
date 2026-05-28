from pathlib import Path

import pandas as pd
from data_analysis.src.logger import logger
from data_analysis.src.morphology import (
    active_present_participle,
    adverbial_participle,
    finite_verb,
    half_participle,
    parse_features,
    participle,
    passive_present_participle,
)

FEATURES = {
    "finite_verb": finite_verb,
    "participle": participle,
    "adverbial_participle": adverbial_participle,
    "half_participle": half_participle,
    "active_present_participle": active_present_participle,
    "passive_present_participle": passive_present_participle,
}

NOUN_CASES = {
    "dative": "Dat",
    "instrumental": "Ins",
}

POS_DEGREES = {
    "ADJ": ["Pos", "Cmp"],
    "ADV": ["Pos", "Cmp"],
}


def mean_sentence_length_from_pos(file_path: Path) -> float:
    """
    Computes mean sentence length directly from .pos file structure.
    Empty lines mark sentence boundaries; punctuation tokens are excluded.
    """
    sentence_lengths = []
    current_length = 0

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                if current_length > 0:
                    sentence_lengths.append(current_length)
                    current_length = 0
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                continue
            if parts[2] == "PUNCT":
                continue

            current_length += 1

        if current_length > 0:
            sentence_lengths.append(current_length)

    if not sentence_lengths:
        return 0.0

    return round(sum(sentence_lengths) / len(sentence_lengths), 1)


def format_cell(freq, pct):
    return f"{freq}\n{pct}%"


def compute_morph_freqs(file_path: Path) -> dict:
    total_tokens = 0
    total_nouns = 0
    total_verbs = 0
    tokens_list = []

    verb_counts = {feat: 0 for feat in FEATURES}
    noun_case_counts = {name: 0 for name in NOUN_CASES}
    pos_totals = {upos: 0 for upos in POS_DEGREES}
    degree_counts = {
        upos: {deg: 0 for deg in degrees} for upos, degrees in POS_DEGREES.items()
    }

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines

            parts = line.split("\t")
            if len(parts) < 5:
                continue

            token_form = parts[0]
            upos = parts[2]
            morph = parts[4]

            if upos == "PUNCT":
                continue

            tokens_list.append(token_form.lower())
            total_tokens += 1

            feats = parse_features(morph)

            if upos == "VERB":
                total_verbs += 1
                for feat, check in FEATURES.items():
                    if check(feats):
                        verb_counts[feat] += 1

            if upos == "NOUN":
                total_nouns += 1
                for name, case_val in NOUN_CASES.items():
                    if feats.get("Case") == case_val:
                        noun_case_counts[name] += 1

            if upos in POS_DEGREES:
                pos_totals[upos] += 1
                degree = feats.get("Degree")
                if degree in degree_counts[upos]:
                    degree_counts[upos][degree] += 1

    ttr = round(len(set(tokens_list)) / total_tokens, 3) if total_tokens else 0.0
    mean_sent_len = mean_sentence_length_from_pos(file_path)

    row = {
        "tokens": total_tokens,
        "types": len(set(tokens_list)),
        "ttr": ttr,
        "mean_sentence_length": mean_sent_len,
        "verbs": total_verbs,
        "nouns": total_nouns,
    }

    for feat, count in verb_counts.items():
        row[f"{feat}"] = round(count / total_tokens * 1000, 1) if total_tokens else 0.0
        row[f"{feat}_%"] = round(count / total_verbs * 100, 1) if total_verbs else 0.0

    for name, count in noun_case_counts.items():
        row[f"{name}"] = round(count / total_tokens * 1000, 1) if total_tokens else 0.0
        row[f"{name}_%"] = round(count / total_nouns * 100, 1) if total_nouns else 0.0

    for upos, degrees in degree_counts.items():
        pos_total = pos_totals[upos]
        for deg, count in degrees.items():
            key = f"{deg.lower()}_{upos.lower()}"
            row[f"{key}"] = (
                round(count / total_tokens * 1000, 1) if total_tokens else 0.0
            )
            row[f"{key}_%"] = round(count / pos_total * 100, 1) if pos_total else 0.0

    return row


def build_condition_table(row: dict):

    table = {
        "tokens": row["tokens"],
        "types": row["types"],
        "ttr": row["ttr"],
        "mean_sentence_length": row["mean_sentence_length"],
        "finite_verb": row["finite_verb_per_1000"],
        "finite_verb_%": row["finite_verb_pct_of_verbs"],
        "participle": row["participle_per_1000"],
        "participle_%": row["participle_pct_of_verbs"],
        "adverbial_participle": row["adverbial_participle_per_1000"],
        "adverbial_participle_%": row["adverbial_participle_pct_of_verbs"],
        "half_participle": row["half_participle_per_1000"],
        "half_participle_%": row["half_participle_pct_of_verbs"],
        "active_present_participle": row["active_present_participle_per_1000"],
        "active_present_participle_%": row["active_present_participle_pct_of_verbs"],
        "passive_present_participle": row["passive_present_participle_per_1000"],
        "passive_present_participle_%": row["passive_present_participle_pct_of_verbs"],
        "dative": row["dative_per_1000"],
        "dative_%": row["dative_pct_of_nouns"],
        "instrumental": row["instrumental_per_1000"],
        "instrumental_%": row["instrumental_pct_of_nouns"],
        "pos_adj": row["pos_adj_per_1000"],
        "pos_adj_%": row["pos_adj_pct_of_adj"],
        "cmp_adj": row["cmp_adj_per_1000"],
        "cmp_adj_%": row["cmp_adj_pct_of_adj"],
        "pos_adv": row["pos_adv_per_1000"],
        "pos_adv_%": row["pos_adv_pct_of_adv"],
        "cmp_adv": row["cmp_adv_per_1000"],
        "cmp_adv_%": row["cmp_adv_pct_of_adv"],
    }

    return pd.DataFrame([table])


def main():
    input_path = Path("data_analysis/data/output/llm/morph_annotations")
    output_dir = Path("data_analysis/data/output/llm/morph_summaries")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for model_dir in sorted(input_path.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        for pos_file in sorted(model_dir.glob("*.pos")):
            # e.g. lithuanian_regular_A1.pos
            parts = pos_file.stem.split("_")
            language = parts[0]
            prompt_type = parts[1]
            cefr = parts[2]

            logger.info(f"Processing: {model_name} / {pos_file.stem}")

            row = compute_morph_freqs(pos_file)
            row["model"] = model_name
            row["language"] = language
            row["type"] = prompt_type
            row["cefr"] = cefr

            all_rows.append(row)

    if not all_rows:
        logger.warning("No .pos files found. Check your input path.")
        return

    df = pd.DataFrame(all_rows)

    # Put metadata columns first
    meta_cols = ["model", "language", "type", "cefr"]
    metric_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + metric_cols]

    df = df.sort_values(["language", "type", "cefr", "model"]).reset_index(drop=True)

    output_path = output_dir / "morph_summary.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
