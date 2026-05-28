from pathlib import Path
import pandas as pd
from settings import CEFR_LEVELS
from data_analysis.src.morphology import (
    active_present_participle,
    adverbial_participle,
    finite_verb,
    half_participle,
    parse_features,
    participle,
    passive_present_participle,
)
from data_analysis.src.logger import logger


POS_DEGREES = {
    "ADJ": ["Pos", "Cmp"],
    "ADV": ["Pos", "Cmp"],
}

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


def compute_sttr(tokens, window_size=1000):
    if len(tokens) < window_size:
        return len(set(tokens)) / len(tokens) if tokens else 0.0

    ttrs = []
    for i in range(0, len(tokens), window_size):
        chunk = tokens[i:i + window_size]
        if len(chunk) < window_size:
            continue
        ttrs.append(len(set(chunk)) / len(chunk))

    return sum(ttrs) / len(ttrs) if ttrs else 0.0


def compute_morph_freqs(folder_path: Path) -> dict:

    total_tokens = 0
    total_nouns = 0
    total_verbs = 0

    tokens_list = []

    verb_counts = {feat: 0 for feat in FEATURES}
    noun_case_counts = {name: 0 for name in NOUN_CASES}
    pos_totals = {upos: 0 for upos in POS_DEGREES}
    degree_counts = {
        upos: {deg: 0 for deg in degrees}
        for upos, degrees in POS_DEGREES.items()
    }

    for file in folder_path.glob("*.pos"):
        with open(file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 5:
                    continue

                upos = parts[2]

                if upos == "PUNCT":
                    continue

                token = parts[0].lower()
                tokens_list.append(token)
                total_tokens += 1

                morph = parts[4]
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

    sttr = round(compute_sttr(tokens_list), 3)

    row = {
        "tokens": total_tokens,
        "verbs": total_verbs,
        "nouns": total_nouns,
        "types": len(set(tokens_list)),
        "sttr": sttr,
    }

    for feat, count in verb_counts.items():
        row[f"{feat}"] = (
            round(count / total_tokens * 1000, 1) if total_tokens else 0.0
        )
        row[f"{feat}_%"] = (
            round(count / total_verbs * 100, 1) if total_verbs else 0.0
        )

    for name, count in noun_case_counts.items():
        row[f"{name}"] = (
            round(count / total_tokens * 1000, 1) if total_tokens else 0.0
        )
        row[f"{name}_%"] = (
            round(count / total_nouns * 100, 1) if total_nouns else 0.0
        )

    for upos, degrees in degree_counts.items():
        pos_total = pos_totals[upos]
        for deg, count in degrees.items():
            key = f"{deg.lower()}_{upos.lower()}"
            row[f"{key}"] = (
                round(count / total_tokens * 1000, 1) if total_tokens else 0.0
            )
            row[f"{key}_%"] = (
                round(count / pos_total * 100, 1) if pos_total else 0.0
            )

    return row

def main():
    output_path = Path(
        "data_analysis/data/output/reference_data/sanity_check"
    )
    input_path = Path(
        "data_analysis/data/output/reference_data/morph_annotations"
    )

    rows = []
    for level in CEFR_LEVELS:
        row = compute_morph_freqs(input_path / level.lower())
        row["level"] = level
        rows.append(row)

    df = pd.DataFrame(rows).set_index("level")

    save_path = output_path / "morph_freqs_final.csv"
    df.to_csv(save_path)

    logger.info(f"Saved to {save_path}")


if __name__ == "__main__":
    main()