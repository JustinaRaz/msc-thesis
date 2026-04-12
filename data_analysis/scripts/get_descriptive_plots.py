from itertools import product
from tqdm import tqdm

from data_analysis.src.data_plotting import Plotter
from data_analysis.src.logger import logger


def main():
    logger.info("Initializing the plotter for text descriptives.")
    plotter = Plotter()

    combinations = list(product(
        ["assistant"],
        ["English", "Lithuanian"],
        ["regular", "constrained"]
    ))

    for role, language, prompt_type in tqdm(
        combinations,
        desc="Generating plots",
        unit="plot"
    ):
        plotter.plot_text_descriptives(role, language, prompt_type)

    logger.info("All plots are saved.")


if __name__ == "__main__":
    main()