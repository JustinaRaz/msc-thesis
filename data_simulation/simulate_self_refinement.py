import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml
from data_simulation.src.data_cleaning import DataCleaner
from data_simulation.src.hf_gemma import ChatHFGemma
from data_simulation.src.model_load import load_model_backend
from data_simulation.src.ref_thresholds import check_thresholds
from data_simulation.src.models import (
    ChatHistory,
    ChatMessage,
    SystemPrompt,
    load_prompt_by_level,
)
from data_simulation.src.ref_evaluation import evaluate
from settings import logger
from tqdm import tqdm

cleaner = DataCleaner()


def input_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt_cefr_level",
        help="CEFR level of prompt.",
        type=str,
        default="A1",
    )
    parser.add_argument(
        "--prompt_language",
        help="Language of prompt to use in the experiment.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_size",
        help="Model size as specified in configs/models.toml",
        type=str,
        default="medium",
    )
    parser.add_argument(
        "--student_constrain",
        help="Include this flag to constrain the student LLM.",
        action="store_true",
    )

    args = parser.parse_args()

    return args


def simulate_conversation(
    model: ChatHFGemma,
    level: str = "A1",
    tutor_system_prompt=SystemPrompt,
    student_system_prompt=SystemPrompt,
    n_total_rounds: int = 9,
    detect_language: bool = True,
    max_regenerations: int = 5,
) -> ChatHistory | None:
    """
    Simulate an LLM conversation using a self-refinement for tutor.
    """

    # Student history (student is assistant)
    student_history = ChatHistory(
        messages=[
            ChatMessage(role="system", content=student_system_prompt.content),
        ]
    )

    # Tutor history (tutor is assistant)
    tutor_history = ChatHistory(
        messages=[
            ChatMessage(role="system", content=tutor_system_prompt.content),
            ChatMessage(
                role="user", content="Labas!"
            ),
        ]
    )

    # Collects the reference texts
    tutor_reference_texts = []

    base_dialogue_window_metrics = None

    # Looping through turns
    for turn_index in tqdm(range(n_total_rounds)):
        logger.info(f"Turn {turn_index + 1}.")

        # Get refinement instructions
        refinement_prompt_file = Path(
             "data_simulation/configs/prompts/refinement.yaml"
        )

        with open(refinement_prompt_file, "r", encoding="utf-8") as file:
            refinement_prompts = yaml.safe_load(file)

        # Tutor turn:
        tutor_message = model.generate(tutor_history)
        tutor_text = tutor_message.content # Extract text
        tutor_text = cleaner.clean_text(tutor_text) # Clean text

        # If it is tutor's 1st or 2nd turn, clean it and append it to the history
        if turn_index in [0, 1]:
            tutor_history.messages.append(
                ChatMessage(role="assistant", content=tutor_text)
            )

        # Collect reference texts
        # If it is tutor's 3rd or 4th turn, append it to reference text list
        elif turn_index in [2, 3]:

            tutor_reference_texts.append(str(tutor_text)) # Append to the list

            tutor_history.messages.append(
                ChatMessage(role="assistant", content=tutor_text)
            )

            # Compute the baseline thresholds:
            if turn_index == 3:
                base_dialogue_window_metrics = evaluate(tutor_reference_texts)
                lower_mdd = base_dialogue_window_metrics["mdd_mean"] - (
                    2 * base_dialogue_window_metrics["mdd_sd"]
                )
                upper_mdd = base_dialogue_window_metrics["mdd_mean"] + (
                    2 * base_dialogue_window_metrics["mdd_sd"]
                )
                lower_len = base_dialogue_window_metrics["avg_text_length"] - (
                    2 * base_dialogue_window_metrics["length_sd"]
                )
                upper_len = base_dialogue_window_metrics["avg_text_length"] + (
                    2 * base_dialogue_window_metrics["length_sd"]
                )

        # Introduce refinement in later turns
        elif turn_index in [4, 5, 6, 7, 8]:

            for attempt in range(max_regenerations):
                metrics = evaluate(tutor_text, refine=True)
                issues = check_thresholds(
                    metrics, lower_mdd, upper_mdd, lower_len, upper_len
                )
                if not issues:
                    tutor_history.messages.append(
                        ChatMessage(role="assistant", content=tutor_text)
                    )
                    logger.info(
                        f"Turn {turn_index + 1} accepted after {attempt + 1} attempt(s)"
                    )
                    break

                feedback = []
                logger.info(f"{len(issues)} issue(s) were found: {issues}.")

                # Constructing feedback
                if "low" in issues:
                    feedback.append(refinement_prompts["mdd"]["low"].format(cefr=level))

                if "high" in issues:
                    feedback.append(
                        refinement_prompts["mdd"]["high"].format(cefr=level)
                    )

                if "short" in issues:
                    feedback.append(
                        refinement_prompts["length"]["short"].format(cefr=level)
                    )

                if "long" in issues:
                    feedback.append(
                        refinement_prompts["length"]["long"].format(cefr=level)
                    )

                temp_history = deepcopy(tutor_history)

                # Append draft into temporary history
                temp_history.messages.append(
                    ChatMessage(role="assistant", content=tutor_text)
                )

                # Append instruction to a tutor
                temp_history.messages.append(
                    ChatMessage(role="user", content=" ".join(feedback))
                )
                
                # Regenerate
                revised_tutor_message = model.generate(temp_history)
                tutor_message = revised_tutor_message
                tutor_text = cleaner.clean_text(revised_tutor_message.content)

            else:
                logger.warning(
                    f"Turn {turn_index + 1}: did not converge after {max_regenerations} attempts, appending the last attempt."
                )
                tutor_history.messages.append(
                    ChatMessage(role="assistant", content=tutor_text)
                )

        # Tutor output becomes student input (user message)
        student_history.messages.append(ChatMessage(role="user", content=tutor_text))

        # Student turn.

        student_message = model.generate(student_history)
        student_history.messages.append(student_message)

        # Student output becomes tutor input (user message)
        tutor_history.messages.append(
            ChatMessage(role="user", content=student_message.content)
        )

    return tutor_history


def main():
    args = input_parse()
    n_runs = 30

    for n in range(n_runs):
        logger.info(f"Running simulation run {n + 1} out of {n_runs}")

        # MODEL LOADING
        sampling_params = {
            "temp": 1,
            "top_p": 1.0,
            "min_p": 0.05,
            "top_k": 50,
        }
        penalty_params = {"repetition_penalty": 1.1}

        cache_dir = Path("data_simulation/models")
        models_config_file = Path("data_simulation/configs/models.toml")

        model = load_model_backend(
            models_config_path=models_config_file,
            model_size=args.model_size,
            token_path=Path("data_simulation/tokens/hf_token.txt"),
            cache_dir=cache_dir,
            sampling_params=sampling_params,
            penalty_params=penalty_params,
        )

        # PROMPT LOADING

        tutor_prompt_file = Path("data_simulation/configs/prompts/tutor.yaml")

        student_prompt_file = Path("data_simulation/configs/prompts/student.yaml")

        logger.info(
            f"Formatting prompts using {args.prompt_language} language "
            f"and CEFR level {args.prompt_cefr_level}"
        )

        tutor_system_prompt = load_prompt_by_level(
            file_path=tutor_prompt_file,
            prompt_level=args.prompt_cefr_level,
            prompt_language=args.prompt_language,
            student_constrain=args.student_constrain,
            role="tutor",
        )

        student_system_prompt = load_prompt_by_level(
            file_path=student_prompt_file,
            prompt_level=args.prompt_cefr_level,
            prompt_language=args.prompt_language,
            student_constrain=args.student_constrain,
            role="student",
        )

        # SIMULATE
        tutor_history = simulate_conversation(
            level=args.prompt_cefr_level,
            model=model,
            n_total_rounds=9,
            tutor_system_prompt=tutor_system_prompt,
            student_system_prompt=student_system_prompt,
        )

        if tutor_history is None:
            logger.info(f"Skipping run {n + 1}")
            del model
            continue

        # SAVE CHAT
        chat_json = json.dumps(
            [msg.model_dump() for msg in tutor_history.messages],
            indent=3,
            ensure_ascii=False,
        )

        save_dir = (
            Path("data_simulation/output/refinement")
            / model.model_id.replace("/", "--")
            / args.prompt_language
            / args.prompt_cefr_level
        )

        save_dir.mkdir(exist_ok=True, parents=True)

        save_file_name = datetime.now().strftime("%m%d-%H%M%S")
        with open(save_dir / f"{save_file_name}.json", "w") as outfile:
            outfile.write(chat_json)

        del model


if __name__ == "__main__":
    main()
