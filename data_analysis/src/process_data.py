from pathlib import Path

import polars as pl
from data_analysis.src.data_cleaning import DataCleaner
from data_analysis.src.logger import logger
from data_analysis.src.models.metadata import DataFile
from settings import MODELS
from tqdm import tqdm

cleaner = DataCleaner()
report = DataCleaner()

class DataProcessor:
    def __init__(self):

        self.base_dir = Path(__file__).parents[2] / "data_simulation" / "output" / "reproducibility"

        self.output_dir = (
            Path(__file__).parents[2] / "data_analysis" / "data" / "output" / "llm" / "llm_text"
        )
        self.clean_report_dir = (
            Path(__file__).parents[2] / "data_analysis" / "data" / "output" / "metrics"
        )

    def collect_metadata(self) -> list[DataFile]:

        logger.info(
            "Collecting metadata of all JSON files with simulated data (LLM dialogues)."
        )

        datafiles = []

        for model_name in MODELS:
            model_path = self.base_dir / model_name

            for top_folder in model_path.iterdir():
                if top_folder.name == "student_constrained":
                    for language_folder in top_folder.iterdir():
                        language = language_folder.name
                        data_type = "constrained"

                        for cefr_folder in language_folder.iterdir():
                            cefr_level = cefr_folder.name

                            for json_file in cefr_folder.glob("*.json"):
                                datafiles.append(
                                    DataFile(
                                        model=model_name,
                                        language=language,
                                        type=data_type,
                                        cefr=cefr_level,
                                        file_name=json_file.name,
                                        path=json_file,
                                    )
                                )

                else:
                    language = top_folder.name
                    data_type = "regular"

                    for cefr_folder in top_folder.iterdir():
                        if not cefr_folder.is_dir():
                            continue

                        cefr_level = cefr_folder.name

                        for json_file in cefr_folder.glob("*.json"):
                            datafiles.append(
                                DataFile(
                                    model=model_name,
                                    language=language,
                                    type=data_type,
                                    cefr=cefr_level,
                                    file_name=json_file.name,
                                    path=json_file,
                                )
                            )

        return datafiles

    def process_data(self) -> pl.DataFrame:

        logger.info("Starting the dataset (conversation data) cleaning and merging into one file.")

        datafiles = self.collect_metadata()

        logger.info(f"Found {len(datafiles)} JSON files.")

        dfs = []

        for file in tqdm(datafiles, desc="Processing JSON files", unit="file"):
            cleaner.set_metadata(file)
            report.set_metadata(file)

            df = pl.read_json(file.path)

            df = df.with_columns(pl.lit(file.file_name).alias("dialogue_id"))

            df = df.with_columns(
                (pl.col("role").eq("assistant").cum_sum().over("dialogue_id")).alias("turn")
            )

            df = df.filter(pl.col("role") != "system")

            # clean all non-system rows for final dataset,
            # but only assistant rows update the report counters
            cleaned_contents = []

            for row in df.iter_rows(named=True):
                text = row["content"]

                if row["role"] == "assistant":
                    cleaned_text = report.clean_text(text)
                else:
                    cleaned_text = cleaner.clean_text(text)

                cleaned_contents.append(cleaned_text)

            df = df.with_columns(
                pl.Series("content", cleaned_contents)
            )

            df = df.with_columns(
                model=pl.lit(file.model),
                language=pl.lit(file.language),
                type=pl.lit(file.type),
                cefr=pl.lit(file.cefr),
            )

            dfs.append(df)

            report.finalize_file()

        df_final = pl.concat(dfs)

        dataset_path = self.output_dir / "clean_dataset.parquet"
        dataset_path_csv = self.output_dir / "clean_dataset.csv"

        df_final.write_parquet(dataset_path)
        df_final.write_csv(dataset_path_csv)

        logger.info(f"Cleaned datasets (in parquet and csv formats) are saved to {self.output_dir}")

        report_path = report.export_report(self.clean_report_dir)

        logger.info(f"Cleaning report saved to {report_path}")

        return df_final
