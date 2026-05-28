from data_analysis.src.process_data import DataProcessor
from data_analysis.src.logger import logger


def main():
    logger.info("Initializing data processor.")
    processor = DataProcessor()

    logger.info("Cleaning the dataset.")
    processor.process_data()


if __name__ == "__main__":
    main()
