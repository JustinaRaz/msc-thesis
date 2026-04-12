from pathlib import Path
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from interact_llm.data_models.chat import ChatMessage

HF_ROLE_MAP = {
    "system": "system",
    "student": "user",
    "tutor": "assistant",
}


class ChatHF:
    """
    Model wrapper for loading and using a HuggingFace causal language model with HF's own libraries
    """

    def __init__(
        self,
        model_id: str,
        cache_dir: Optional[Path] = None,
        sampling_params: Optional[dict] = None,
        penalty_params: Optional[dict] = None,
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.sampling_params = sampling_params
        self.penalty_params = penalty_params

    def load(self) -> None:
        """
        Load the model and the tokenizer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, self.cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            cache_dir = self.cache_dir,
            torch_dtype = "auto",
            device_map = "auto",
        )

    def format_params(self) -> dict:
        params = {}

        if self.sampling_params:
            params.update(self.sampling_params)

        if self.penalty_params:
            params.update(self.penalty_params)

        return params


    def generate(self, chat: list[ChatMessage], max_new_tokens: int = 3000):

        kwargs = self.format_params()
        do_sample = bool(kwargs)

        self.tokenizer.use_default_system_prompt = False

        hf_chat = [
            {
                "role": HF_ROLE_MAP[m.role],
                "content": m.content,
            }
            for m in chat
        ]

        text = self.tokenizer.apply_chat_template(
            hf_chat,
            tokenize = False,
            add_generation_prompt = True,
        )

        model_inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        input_len = model_inputs["input_ids"].shape[-1]

        output = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **kwargs,
        )

        # chat (decoded output)
        response = self.tokenizer.decode(
            output[0][input_len:], skip_special_tokens=True
        )

        chat_message = ChatMessage(role = "tutor", content = response)

        return chat_message
