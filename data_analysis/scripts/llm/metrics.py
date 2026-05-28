from data_analysis.src.metrics import MetricExtractor
from data_analysis.src.logger import logger


data_file = "data_analysis/output/llm/llm_text/clean_dataset.parquet"
output_file = "data_analysis/output/metrics/llm_data_metrics.parquet"

def main():
    logger.info("Initializing metric extraction.")

    extractor = MetricExtractor(
    data_file=data_file,
    output_file=output_file,
)

    logger.info("Extracting text descriptives.")
    extractor.get_text_descriptives()

if __name__ == "__main__":
    main()