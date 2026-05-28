import logging

logger = logging.getLogger("msc-thesis")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.propagate = False

MODELS = [
    "google--gemma-3-4b-it",
    "google--gemma-3-12b-it",
    "google--gemma-3-27b-it",
]

MODEL_NAMES = [
        "Gemma3 4B IT",
        "Gemma3 12B IT",
        "Gemma3 27B IT",
    ]

CEFR_LEVELS = ["A1", "B1", "C1"]

MODEL_COLORS = {
        "google--gemma-3-4b-it": "olivedrab",
        "google--gemma-3-12b-it": "gold",
        "google--gemma-3-27b-it": "darkred",
    }

LANGUAGE_MAPPING = {
    "EN": "English", 
    "LT": "Lithuanian"
    }

MODEL_NAMING = {
    "google--gemma-3-4b-it": "Gemma 3 4B",
    "google--gemma-3-12b-it": "Gemma 3 12B",
    "google--gemma-3-27b-it": "Gemma 3 27B",
}


CEFR_COLORS = {
            "A1": "olivedrab",
            "B1": "gold",
            "C1": "darkred",
        }

CEFR_GREY_COLORS = {
            "A1": "darkgrey",
            "B1": "dimgrey",
            "C1": "darkslategrey",
        }