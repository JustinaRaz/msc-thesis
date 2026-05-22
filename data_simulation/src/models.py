from pathlib import Path
from typing import Literal

import yaml
from settings import logger
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """
    Chat message formatting.

    Args:
        role: the sender of the content.
            user = input
            assistant = LLM output
            system = initial system message only

        content: text written by the role.
    """

    role: Literal["user", "assistant", "system"]
    content: str


class ChatHistory(BaseModel):
    """
    Chat history formatting.
    """

    messages: list[ChatMessage]


class Prompt(BaseModel):
    id: str
    content: str


class SystemPrompt(Prompt):
    role: str = Field(default="system", frozen=True)


def load_prompt_by_level(
    file_path: Path,
    prompt_level: str,
    prompt_language: str,
    student_constrain: bool,
    system_prompt: bool = True,
    role: Literal["tutor", "student"] = "tutor",
) -> Prompt | SystemPrompt:
    """
    Load a prompt by CEFR LEVEL from a YAML file and return it as either a SystemPrompt or a regular Prompt.

    Args:
        file_path (Path): Path to one of the YAML file.
        prompt_leve (str): The CEFR LEVEL of the prompt to retrieve.
        prompt_language (str): The language the prompt should be loaded in.
        student_constrain (bool): If true, it constrains student LLM to CEFR level.
        system_prompt (bool): Whether to return a SystemPrompt or a regular Prompt.

    Returns:
        Prompt | SystemPrompt: The requested prompt if found.
    """

    with open(file_path, "r", encoding="utf-8") as file:
        prompts = yaml.safe_load(file)

    try:
        if not student_constrain and role == "student":
            prompt_text = prompts["student_prompts"][prompt_language]
        else:
            prompt_text = prompts[f"{role}_prompts"][prompt_level][prompt_language]

        prompt_class = SystemPrompt if system_prompt else Prompt
        return prompt_class(id=prompt_level, content=prompt_text)

    except KeyError:
        logger.error(
            f"Error: Prompt for level {prompt_level} in {prompt_language} not found."
        )
        return None

    logger.warning(
        f"[WARNING:] No prompt found with LEVEL {prompt_level}, running without custom prompt."
    )
    return None
