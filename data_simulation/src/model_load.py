from pathlib import Path
from typing import Optional

import toml
from data_simulation.src.hf_gemma import ChatHFGemma
from huggingface_hub import login
from settings import logger


def get_model_id(models_config_path: Path, model_size: str) -> str:
    """
    Get model ID based on the model size [small, medium, large].
    """

    models = toml.load(models_config_path)["models"]

    for model in models:
        if model.get("size") == model_size:
            return model["id"]


def login_hf_token(
    token_path: Path("data_simulation/tokens/hf_token.txt")
) -> None:

    try:
        with open(token_path) as f:
            hf_token = f.read().strip()

        login(hf_token)
        logger.info("Logged in to Hugging Face successfully.")

    except Exception as e:
        logger.error(f"Error during Hugging Face login: {e}")
        raise


def load_model_backend(
    models_config_path: Path,
    model_size: str,
    token_path: Path("data_simulation/tokens/hf_token.txt"),
    cache_dir: Optional[Path] = None,
    **model_kwargs,
) -> ChatHFGemma:

    model_id = get_model_id(models_config_path, model_size)

    model = ChatHFGemma(
        model_id=model_id,
        cache_dir=cache_dir,
        **model_kwargs,
    )

    try:
        model.load()
        logger.info(f"Gemma {model_size} loaded successfully.")

    except OSError as e:
        if "401 Client Error" in str(e):
            logger.error("Authentication error. Logging into Hugging Face...")

            login_hf_token(token_path)

            model.load()
            logger.info(f"Gemma {model_size} loaded successfully after the login.")
        else:
            raise

    return model
