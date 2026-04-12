import polars as pl
from pathlib import Path
from tqdm import tqdm

from data_analysis.utils.models.metadata import DataFile
from data_analysis.utils.data_cleaning import DataCleaner
from data_analysis.utils.logger import logger


cleaner = DataCleaner()


class DataProcessor:

    def __init__(self):

        self.model_dirs = [
            "google--gemma-3-4b-it",
            "google--gemma-3-12b-it",
            "google--gemma-3-27b-it",
        ]

        self.base_dir = (
            Path(__file__).parents[2] / "interact-llm" / "simulated_data"
        )

        self.output_dir = Path(__file__).parents[2] / "data_analysis" / "data" / "llm_text"
        self.clean_report_dir = Path(__file__).parents[2] / "data_analysis" / "data" / "metrics"

    def collect_metadata(self) -> list[DataFile]:

        logger.info(
            "Collecting metadata of all JSON files with simulated data (LLM dialogues)."
        )

        datafiles = []

        for model_name in self.model_dirs:

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

        logger.info("Starting the dataset cleaning and merging.")

        datafiles = self.collect_metadata()

        logger.info(f"Found {len(datafiles)} JSON files.")

        dfs = []

        for file in tqdm(datafiles, desc="Processing JSON files", unit="file"):

            cleaner.set_metadata(file)

            df = pl.read_json(file.path)

            df = df.with_columns(
                pl.lit(file.file_name).alias("dialogue_id")
            )

            df = df.with_columns(
                (
                    pl.col("role")
                    .eq("assistant")
                    .cum_sum()
                    .over("dialogue_id")
                ).alias("turn")
            )

            df = df.filter(pl.col("role") == "assistant")

            df = df.with_columns(
                pl.col("content").map_elements(
                    cleaner.clean_text,
                    return_dtype=pl.String,
                )
            )

            df = df.with_columns(
                model=pl.lit(file.model),
                language=pl.lit(file.language),
                type=pl.lit(file.type),
                cefr=pl.lit(file.cefr)
                #id=pl.lit(file.file_name),
            )

            dfs.append(df)

            cleaner.finalize_file()

        df_final = pl.concat(dfs)

        dataset_path = self.output_dir / "clean_dataset.parquet"

        df_final.write_parquet(dataset_path)

        logger.info(f"Dataset saved to {dataset_path}")

        report_path = cleaner.export_report(self.clean_report_dir)

        logger.info(f"Cleaning report saved to {report_path}")

        return df_final 

        