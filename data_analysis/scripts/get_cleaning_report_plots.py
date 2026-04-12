from itertools import product
from tqdm import tqdm
import polars as pl
from data_analysis.src.data_plotting import Plotter
from data_analysis.src.logger import logger


def main():
    logger.info("Initializing the plotter.")
    plotter = Plotter()

    data = pl.read_csv("data_analysis/data/cleaning_report.csv")

#     data_result = (
#     data
#     .group_by(["model", "prompt_type", "language", "cefr"])
#     .agg([
#         pl.col("repetitive_words_removed").sum().alias("total_repetitive_words_removed"),
#         pl.col("invalid_sentences_removed").sum().alias("total_invalid_sentences_removed"),
#         pl.col("long_words_removed").sum().alias("total_long_words_removed"),
#         pl.col("invalid_char_words_removed").sum().alias("total_invalid_char_words_removed"),
#     ])
#     .sort(["model", "prompt_type", "language", "cefr"])
# )

    plotter.plot_cleaning_report(data)

    logger.info("All plots are saved.")


if __name__ == "__main__":
    main()