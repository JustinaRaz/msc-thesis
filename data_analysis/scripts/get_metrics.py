from data_analysis.src.metrics import MetricExtractor
from data_analysis.src.logger import logger


def main():
    logger.info("Initializing metric extraction.")
    extractor = MetricExtractor()

    logger.info("Extracting text descriptives.")
    extractor.get_text_descriptives()

if __name__ == "__main__":
    main()