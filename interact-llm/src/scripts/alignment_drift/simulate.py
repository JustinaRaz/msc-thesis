"""
Simulate two chat LLMs talking to each other
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from interact_llm.data_models.chat import ChatHistory, ChatMessage
from interact_llm.data_models.prompt import SystemPrompt, load_prompt_by_level
from interact_llm.llm.hf_wrapper import ChatHF
from interact_llm.llm.hf_gemma import ChatHFGemma
from interact_llm.utils.model_load import load_model_backend
from scripts.alignment_drift.detect_lang import _detect_lang

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
        default="small",
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
    tutor_system_prompt = SystemPrompt,
    student_system_prompt = SystemPrompt,
    n_total_rounds: int = 9,
    detect_language: bool = True,
) -> ChatHistory | None:
    """
    Simulate an LLM conversation.

    Note:
    - tutor_history: model speaks as assistant
    - student_history: model also speaks as assistant
    - each side receives the other side's output as a user message
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
            ChatMessage(role="user", content="Labas!"), # Fixed first message to the tutor LLM
        ]
    )

    for _ in tqdm(range(n_total_rounds)):

        # Tutor turn.

        tutor_message = model.generate(tutor_history)
        tutor_history.messages.append(tutor_message)

        # Tutor output becomes student input (user message)
        student_history.messages.append(
            ChatMessage(role="user", content=tutor_message.content)
        )

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
    n_runs = 3

    for n in range(n_runs):
        print(f"[INFO]: Running simulation run {n + 1} out of {n_runs}")

        # MODEL LOADING
        sampling_params = {
            "temp": 1,
            "top_p": 1.0,
            "min_p": 0.05,
            "top_k": 50,
        }
        penalty_params = {"repetition_penalty": 1.1}

        cache_dir = Path(__file__).parents[3] / "models"
        models_config_file = Path(__file__).parents[3] / "configs" / "models.toml"

        model = load_model_backend(
            models_config_path=models_config_file,
            model_size=args.model_size,
            token_path=Path(__file__).parents[3] / "tokens" / "hf_token.txt",
            cache_dir=cache_dir,
            sampling_params=sampling_params,
            penalty_params=penalty_params,
        )

        # PROMPT LOADING

        tutor_prompt_file = (
            Path(__file__).parents[3]
            / "configs"
            / "prompts"
            / f"tutor.yaml"
        )

        student_prompt_file = (
            Path(__file__).parents[3]
            / "configs"
            / "prompts"
            / f"student.yaml"
        )

        print(
            f"[INFO]: Formatting prompts using {args.prompt_language} language "
            f"and CEFR level {args.prompt_cefr_level}"
        )

        tutor_system_prompt = load_prompt_by_level(
            file_path=tutor_prompt_file,
            prompt_level=args.prompt_cefr_level,
            prompt_language=args.prompt_language,
            student_constrain=args.student_constrain,
            role="tutor"

        )

        student_system_prompt = load_prompt_by_level(
            file_path=student_prompt_file,
            prompt_level=args.prompt_cefr_level,
            prompt_language=args.prompt_language,
            student_constrain=args.student_constrain,
            role="student"

        )


        # SIMULATE
        tutor_history = simulate_conversation(
            model=model,
            n_total_rounds=9,
            tutor_system_prompt=tutor_system_prompt,
            student_system_prompt=student_system_prompt,
        )

        if tutor_history is None:
            print(f"[INFO]: Skipping run {n + 1}")
            del model
            continue

        # SAVE CHAT
        chat_json = json.dumps(
            [msg.model_dump() for msg in tutor_history.messages],
            indent=3,
            ensure_ascii=False,
        )

        save_dir = (
            Path(__file__).parents[4]
            / "interact-llm"
            / "simulated_data"
            / model.model_id.replace("/", "--")
            / f"{args.prompt_language}"
            / args.prompt_cefr_level
        )

        if args.student_constrain:
            save_dir = (
            Path(__file__).parents[4]
            / "interact-llm"
            / "simulated_data"
            / model.model_id.replace("/", "--")
            / "student_constrained"
            / f"{args.prompt_language}"
            / args.prompt_cefr_level
        )

        save_dir.mkdir(exist_ok=True, parents=True)

        save_file_name = datetime.now().strftime("%m%d-%H%M%S")
        with open(save_dir / f"{save_file_name}.json", "w") as outfile:
            outfile.write(chat_json)

        del model


if __name__ == "__main__":
    main()
