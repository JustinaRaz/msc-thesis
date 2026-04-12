from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
from data_analysis.src.logger import logger
from settings import CEFR_LEVELS, MODELS

FORMULAS = {
    "rq1": "{metric} ~ level",
    "rq2": "{metric} ~ level * language",
    "rq3": "{metric} ~ level * type",
}

METRICS = ["dependency_distance_mean", "n_tokens", "sentence_length_mean"]

ROLE = "assistant"

PATHS = {
    "input": Path("data_analysis/data/metrics.csv"),
    "output": Path("data_analysis/data/mixed_effects"),
}


def significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def apply_bonferroni(results: pd.DataFrame) -> pd.DataFrame:
    total_tests = len(results)
    results["p_value_corrected"] = (results["p_value"] * total_tests).clip(upper=1.0)
    results["stars"] = results["p_value_corrected"].apply(significance_stars)
    return results


def filter_data(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    for col, val in cfg.items():
        out = out[out[col] == val]
    return out.reset_index(drop=True)


def fit_one_metric(
    data: pd.DataFrame,
    metric: str,
    model_name: str,
    formula_template: str,
) -> list[dict]:

    model_data = (
        data[data["model"] == model_name]
        .dropna(subset=[metric])
        .reset_index(drop=True)
    )

    logger.info(f"Fitting {metric} for {model_name}")

    formula = formula_template.format(metric=metric)

    fit = smf.mixedlm(
        formula,
        model_data,
        groups=model_data["dialogue_id"]
    ).fit(reml=True)

    return [
        {
            "model": model_name,
            "metric": metric,
            "term": term,
            "estimate": fit.params[term],
            "std_error": fit.bse[term],
            "t_value": fit.tvalues[term],
            "p_value": fit.pvalues[term],
        }
        for term in fit.params.index
    ]


def fit_mixed_models(data, metrics, models, formula_template):
    rows = []

    for model_name in models:
        for metric in metrics:
            rows.extend(
                fit_one_metric(
                    data=data,
                    metric=metric,
                    model_name=model_name,
                    formula_template=formula_template,
                )
            )

    return pd.DataFrame(rows)


# -------------------------
# Load data
# -------------------------

def load_data(path: Path, role: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["role"] == role].copy()

    df["model"] = df["model"].astype("category")
    df["language"] = df["language"].astype("category")
    df["type"] = df["type"].astype("category")

    df = df.rename(columns={"cefr": "level"})
    df["level"] = pd.Categorical(df["level"], categories=CEFR_LEVELS, ordered=True)

    return df


# -------------------------
# Main
# -------------------------

def main():
    logger.info("Starting mixed-effects analysis")

    output_dir = PATHS["output"]
    output_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_data(PATHS["input"], ROLE)

    # -------------------------
    # RQ1
    # -------------------------
    logger.info("Running RQ1")

    df_rq1 = filter_data(df_raw, {
        "type": "regular",
        "language": "English"
    })

    results = fit_mixed_models(
        data=df_rq1,
        metrics=METRICS,
        models=MODELS,
        formula_template=FORMULAS["rq1"],
    )

    results = apply_bonferroni(results)

    out_path = output_dir / "RQ1_results.csv"
    results.to_csv(out_path, index=False)
    logger.info(f"Saved RQ1 results to {out_path}")


    # -------------------------
    # RQ2
    # -------------------------
    logger.info("Running RQ2")

    df_rq2 = filter_data(df_raw, {
        "type": "regular"
        # NOTE: language NOT filtered anymore
    })

    results = fit_mixed_models(
        data=df_rq2,
        metrics=METRICS,
        models=MODELS,
        formula_template=FORMULAS["rq2"],
    )

    results = apply_bonferroni(results)

    out_path = output_dir / "RQ2_results.csv"
    results.to_csv(out_path, index=False)
    logger.info(f"Saved RQ2 results to {out_path}")


    # -------------------------
    # RQ3 (two datasets)
    # -------------------------
    logger.info("Running RQ3")

    for lang in ["English", "Lithuanian"]:

        df_rq3 = filter_data(df_raw, {
            "language": lang
            # NOTE: type NOT filtered anymore
        })

        results = fit_mixed_models(
            data=df_rq3,
            metrics=METRICS,
            models=MODELS,
            formula_template=FORMULAS["rq3"],
        )

        results = apply_bonferroni(results)

        out_path = output_dir / f"RQ3_{lang.upper()}_results.csv"
        results.to_csv(out_path, index=False)

        logger.info(f"Saved RQ3 {lang} results to {out_path}")


if __name__ == "__main__":
    main()
