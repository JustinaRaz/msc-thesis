"""
Utils for model loading with Hugging Face backend only
"""

from pathlib import Path
from typing import Optional

import toml

from interact_llm.llm.hf_wrapper import ChatHF
from interact_llm.llm.hf_gemma import ChatHFGemma
#from interact_llm.utils.model_load.logger import log


def get_model_id(models_config_path: Path, model_size: str) -> str:

    """
    Reads configs/models.toml file and returns the correct model ID based on the specified Gemma model size.
    """

    models = toml.load(models_config_path)["models"]

    for model in models:
        if model.get("size") == model_size:
            return model["hf"]


def login_hf_token(
    token_path: Path = Path(__file__).parents[3] / "tokens" / "hf_token.txt"
) -> None:
    from huggingface_hub import login

    try:
        with open(token_path) as f:
            hf_token = f.read().strip()

        login(hf_token)
        print("Logged in to Hugging Face successfully.")

    except Exception as e:
        print(f"Error during Hugging Face login: {e}")
        raise


def load_model_backend(
    models_config_path: Path,
    model_size: str,
    token_path: Path = Path(__file__).parents[3] / "tokens" / "hf_token.txt",
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
        print(f"Model Gemma {model_size} loaded successfully (model_id = {model_id}).")

    except OSError as e:
        if "401 Client Error" in str(e):
            print("Authentication error. Logging into Hugging Face...")

            login_hf_token(token_path)

            model.load()
            print(f"Model Gemma {model_size} loaded successfully after login.")
        else:
            raise

    return model

