import numpy as np
import pandas as pd
import spacy
import textdescriptives as td

nlp = spacy.load("lt_core_news_md")


def evaluate(text, refine=False):
    """
    text: str if refine=True, list[str] if refine=False
    returns: dict with MDD + length + SDs
    """

    if refine:
        text = [text]

    df = td.extract_metrics(
        text=text, spacy_model="lt_core_news_md", metrics=["dependency_distance"]
    )

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    mdd_mean = df["dependency_distance_mean"].mean()

    docs = list(nlp.pipe(text))
    doc_lengths = [len(doc) for doc in docs]
    len_mean = np.mean(doc_lengths)

    if refine:
        return {
            "mdd_mean": float(mdd_mean),
            "avg_text_length": float(len_mean),
        }

    mdd_sd = df["dependency_distance_mean"].std(ddof=0)
    len_sd = np.std(doc_lengths, ddof=0)

    return {
        "mdd_mean": float(mdd_mean),
        "mdd_sd": float(mdd_sd),
        "avg_text_length": float(len_mean),
        "length_sd": float(len_sd),
    }
