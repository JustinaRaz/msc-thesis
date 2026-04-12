import logging

logger = logging.getLogger("data-analysis")

logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.propagate = False