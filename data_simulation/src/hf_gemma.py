from pathlib import Path
from typing import Optional

import torch
from data_simulation.src.models import ChatMessage
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


class ChatHFGemma:
    """
    Model wrapper for loading and using a HuggingFace causal language model with HF's own libraries
    """

    def __init__(
        self,
        model_id: str = "google/gemma-3-4b-it",
        cache_dir: Optional[Path] = None,
        sampling_params: Optional[dict] = None,
        penalty_params: Optional[dict] = None,
        max_memory: Optional[dict] = {0: "48GB"},
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.processor = None
        self.model = None
        self.sampling_params = sampling_params
        self.penalty_params = penalty_params
        self.max_memory = max_memory

    def load(self) -> None:
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, cache_dir=self.cache_dir, use_fast=True
            )

        if self.model is None:
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                cache_dir=self.cache_dir,
                max_memory=self.max_memory,
            ).eval()

    def format_params(self):
        if self.sampling_params:
            if "temp" in self.sampling_params:
                self.sampling_params["temperature"] = self.sampling_params.pop("temp")

            kwargs = self.sampling_params
        else:
            kwargs = {}

        if self.penalty_params:
            kwargs.update(self.penalty_params)

        return kwargs

    def format_chat_for_gemma(self, chat: list[ChatMessage]) -> list[dict]:
        formatted_chat = []

        for msg in chat.messages:
            formatted_chat.append(
                {"role": msg.role, "content": [{"type": "text", "text": msg.content}]}
            )

        return formatted_chat

    def generate(self, chat: list[ChatMessage], max_new_tokens: int = 3000):
        kwargs = self.format_params()

        if len(kwargs) > 0:
            do_sample = True
        else:
            do_sample = False
            print(
                "[INFO:] No sampling parameters nor penalty parameters were passed. Setting do_sample to 'False'"
            )

        self.processor.use_default_system_prompt = False

        formatted_chat = self.format_chat_for_gemma(chat)

        model_inputs = self.processor.apply_chat_template(
            formatted_chat,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding="longest",
            pad_to_multiple_of=8,
        ).to(self.model.device, dtype=torch.bfloat16)

        self.processor.tokenizer.padding_side = "left"

        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                **kwargs,
            )

        response = self.processor.decode(
            output[0][input_len:], skip_special_tokens=True
        )

        chat_message = ChatMessage(role="assistant", content=response)

        return chat_message
