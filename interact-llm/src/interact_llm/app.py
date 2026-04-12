"""
Initial inspiration:
https://textual.textualize.io/blog/2024/09/15/anatomy-of-a-textual-user-interface/#were-in-the-pipe-five-by-five
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Grid, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, Label, Markdown
from transformers.utils.logging import disable_progress_bar

from .data_models.chat import ChatHistory, ChatMessage
from .data_models.prompt import load_prompt_by_id
from .llm.hf_wrapper import ChatHF

disable_progress_bar()

DEFAULT_PROMPT_VERSION = 3.0


# =========================
# CLI ARGUMENTS
# =========================

def input_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_id", type=str, default="A1")
    parser.add_argument("--prompt_version", type=float, default=DEFAULT_PROMPT_VERSION)
    return parser.parse_args()


# =========================
# QUIT DIALOG
# =========================

class QuitScreen(ModalScreen[bool]):

    def compose(self) -> ComposeResult:
        yield Grid(
            Label("Are you sure you want to quit?", id="question"),
            Button("Quit", variant="error", id="quit"),
            Button("Cancel", variant="primary", id="cancel"),
            id="dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "quit")


# =========================
# MESSAGE FORMATTING
# =========================

class UserMessage(Markdown):
    pass


class Response(Markdown):
    BORDER_TITLE = "Interact-LLM"


# =========================
# CHAT APP
# =========================

class ChatApp(App):

    AUTO_FOCUS = "INPUT"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [("q", "request_quit", "Quit")]

    CSS = """
    UserMessage {
        background: $primary 10%;
        margin: 1;
        margin-right: 8;
        padding: 1 2 0 2;
    }

    Response {
        border: wide $success;
        background: $success 10%;
        margin: 1;
        margin-left: 8;
        padding: 1 2 0 2;
    }

    QuitScreen {
        align: center middle;
    }

    #dialog {
        grid-size: 2;
        grid-gutter: 1 2;
        grid-rows: 1fr 3;
        padding: 0 1;
        width: 60;
        height: 11;
        border: thick $background 80%;
        background: $surface;
    }

    #question {
        column-span: 2;
        content-align: center middle;
    }

    Button {
        width: 100%;
    }
    """

    def __init__(
        self,
        model: ChatHF,
        chat_history: Optional[ChatHistory] = None,
        chat_messages_dir: Optional[Path] = None,
    ):
        super().__init__()
        self.model = model
        self.chat_history = chat_history or ChatHistory(messages=[])
        self.chat_messages_dir = chat_messages_dir

        if self.model.model is None:
            self.exit("[ERROR] Model not loaded.")

        if self.chat_messages_dir:
            self.chat_messages_dir.mkdir(parents=True, exist_ok=True)

    def update_chat_history(self, msg: ChatMessage):
        self.chat_history.messages.append(msg)

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="chat-view"):
            yield Response("Sveiki, ar norite praktikuotis su manimi?")
        yield Input(placeholder="Įrašykite savo žinutę čia")
        yield Footer()

    def action_request_quit(self) -> None:

        def check_quit(quit: bool | None):
            if quit and self.chat_messages_dir:
                chat_json = json.dumps(
                    [msg.dict() for msg in self.chat_history.messages],
                    indent=2,
                    ensure_ascii=False,
                )
                fname = datetime.now().strftime("%Y%m%d-%H%M%S")
                with open(self.chat_messages_dir / f"{fname}.json", "w") as f:
                    f.write(chat_json)
            self.exit()

        self.push_screen(QuitScreen(), check_quit)

    @on(Input.Submitted)
    async def on_input(self, event: Input.Submitted):
        chat_view = self.query_one("#chat-view")
        event.input.clear()

        await chat_view.mount(UserMessage(event.value))
        await chat_view.mount(response := Response())
        response.anchor()

        self.get_model_response(event.value, response)

    @work(thread=True)
    def get_model_response(self, user_text: str, response: Response):
        self.update_chat_history(ChatMessage(role="user", content=user_text))

        model_response = self.model.generate(self.chat_history)
        model_response.content = model_response.content.replace("<|im_end|>", "")

        text = ""
        for chunk in model_response.content:
            text += chunk
            self.call_from_thread(response.update, text)

        self.update_chat_history(model_response)


# =========================
# MAIN
# =========================

def main():
    args = input_parse()

    prompt_file = (
        Path(__file__).parents[2]
        / "configs"
        / "prompts"
        / f"v{args.prompt_version}.toml"
    )

    system_prompt = load_prompt_by_id(
        toml_path=prompt_file,
        prompt_id=args.prompt_id,
        system_prompt=True,
    )

    chat_history = ChatHistory(
        messages=[ChatMessage(role=system_prompt.role, content=system_prompt.content)]
    )

    sampling_params = {"temp": 0.8, "top_p": 0.95, "top_k": 40}
    penalty_params = {"repetition_penalty": 1.1}

    model_id = "BSC-LT/salamandra-2b-instruct"
    cache_dir = Path(__file__).parents[3] / "models"

    print(f"[INFO] Loading model {model_id} with HuggingFace...")

    model = ChatHF(
        model_id=model_id,
        cache_dir=cache_dir,
        sampling_params=sampling_params,
        penalty_params=penalty_params,
    )
    model.load()

    save_dir = (
        Path(__file__).parents[3]
        / "data"
        / model_id.replace("/", "--")
        / f"v{args.prompt_version}"
        / args.prompt_id
    )

    app = ChatApp(
        model=model,
        chat_history=chat_history,
        chat_messages_dir=save_dir,
    )
    app.run()


if __name__ == "__main__":
    main()

