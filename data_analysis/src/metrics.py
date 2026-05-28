import spacy
import textdescriptives as td
import polars as pl
from data_analysis.src.logger import logger
from pathlib import Path
from tqdm import tqdm
from data_analysis.src.data_cleaning import DataCleaner
from hunspell import HunSpell

class MetricExtractor:

    def __init__(
        self,
        data_file: str | Path,
        output_file: str | Path | None = None,
    ):
        self.data_file = Path(data_file)
        Path(output_file) if output_file is not None else None

        self.spacy_model = "lt_core_news_md"
        self.hunspell_obj = HunSpell(
            '/usr/share/hunspell/lt_LT.dic',
            '/usr/share/hunspell/lt_LT.aff'
        )
        self.cleaner = DataCleaner()

    def load_data(self) -> pl.DataFrame:
        df = pl.read_parquet(self.data_file)
        return df

    def extract_textdescriptives(
        self,
        df: pl.DataFrame,
        metrics: list[str] = [
            "descriptive_stats",
            "dependency_distance",
            "pos_proportions",
            "quality",
            "coherence",
        ],
        batch_size: int = 1000,
    ) -> pl.DataFrame:

        logger.info(f"Loading spacy model {self.spacy_model}.")
        model = spacy.load(f"{self.spacy_model}") #python -m spacy download lt_core_news_md

        logger.info("Extracting TextDescriptives.")

        texts = df["content"].to_list()

        results = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Computing metrics"):

            batch = texts[i:i + batch_size]

            batch_metrics = td.extract_metrics(
                text=batch,
                spacy_model=self.spacy_model,
                metrics=metrics,
            )

            results.append(pl.from_pandas(batch_metrics))

        metrics_df = pl.concat(results)

        metrics_df = metrics_df.drop("text")

        combined_df = pl.concat([df, metrics_df], how="horizontal")

        combined_df.write_parquet(self.output_file)

        logger.info(f"Metrics saved at {self.output_file}.")

        return combined_df

    def get_text_descriptives(self) -> pl.DataFrame:

        logger.info("Loading the data.")

        df = self.load_data()

        logger.info(f"Processing {df.shape[0]} rows.")

        df = self.extract_textdescriptives(df)


    def sentence_error_rate(self, sentence):

        tokens = self.cleaner.tokenize(sentence)

        if not tokens:
            return 0.0

        errors = 0

        for token in tokens:
            try:
                token_encoded = token.encode("ISO-8859-13")
                is_correct = self.hunspell_obj.spell(token_encoded)
            except UnicodeEncodeError:
                is_correct = False

            if not is_correct:
                errors += 1
        error_rate = errors / len(tokens)

        return error_rate

    def compute_sentence_error_rates(self):
        
        logger.info("Loading the data.")
        df = self.load_data()

        df = df.filter(
            (pl.col("role") == "assistant"))

        results = []

        for row in df.iter_rows(named=True):

            text = row["content"]

            sentences = self.cleaner.split_to_sentences(text, keep_punctuation = False)

            for sentence in sentences:

                sentence = sentence.strip()

                if not sentence:
                    continue

                error_rate = self.sentence_error_rate(sentence)

                results.append({
                    "dialogue_id": row.get("dialogue_id"),
                    "model": row.get("model"),
                    "cefr": row.get("cefr"),
                    "language": row.get("language"),
                    "type": row.get("type"),
                    "sentence": sentence,
                    "sentence_error_rate": error_rate
                })

        return pl.DataFrame(results)

