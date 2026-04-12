from itertools import product
from tqdm import tqdm

from data_analysis.src.data_plotting import Plotter
from data_analysis.src.logger import logger

from data_analysis.src.get_metrics import MetricExtractor
import polars as pl


def main():

    extractor = MetricExtractor()
    plotter = Plotter()

    df = extractor.compute_sentence_error_rates()

    logger.info("Computing average error rate per dialogue.")

    dialogue_error = (
    df
    .group_by(["dialogue_id", "model", "language", "cefr", "type"])
    .agg(
        pl.col("sentence_error_rate").mean().alias("dialogue_error_rate")
    )
    )

    condition_error = (
    dialogue_error
    .group_by(["model", "language", "cefr", "type"])
    .agg([
        pl.col("dialogue_error_rate").mean().alias("mean_error"),
        pl.col("dialogue_error_rate").std().alias("std_error"),
        pl.count().alias("n_dialogues")
    ])
    .with_columns(
        (1.96 * pl.col("std_error") / pl.col("n_dialogues").sqrt())
        .alias("ci_95")
    )
    .sort(["language", "type", "model", "cefr"])
    )

    df_plot = condition_error.to_pandas()

    plotter.plot_error_rates(df_plot)

    logger.info("Plot is saved.")


if __name__ == "__main__":
    main()