import math

import mlx.core as mx
import numpy as np

from mflux.models.common.tokenizer import Tokenizer
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder


class Flux3PromptEncoder:
    OUTPUT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
    MAX_LENGTH = 4096
    PAD_MULTIPLE = 80

    @staticmethod
    def tokenize(prompt: str, tokenizer: Tokenizer) -> tuple[mx.array, mx.array]:
        # Qwen3-VL chat template, then right padding to the next multiple of 80 (reference behaviour).
        hf = tokenizer.tokenizer
        try:
            text = hf.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        cap = Flux3PromptEncoder.MAX_LENGTH
        pad_mult = Flux3PromptEncoder.PAD_MULTIPLE
        try:
            real = len(hf(text, truncation=True, max_length=cap)["input_ids"])
        except Exception:
            real = pad_mult
        length = max(pad_mult, min(math.ceil(real / pad_mult) * pad_mult, cap))
        tokens = hf(text, return_tensors="np", padding="max_length", truncation=True, max_length=length)
        input_ids = np.asarray(tokens["input_ids"])
        if input_ids.ndim == 0:
            input_ids = input_ids.reshape(1, 1)
        elif input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)

        raw_mask = tokens.get("attention_mask") if isinstance(tokens, dict) else getattr(tokens, "attention_mask", None)
        if raw_mask is not None:
            attention_mask = np.asarray(raw_mask)
            if attention_mask.ndim == 0:
                attention_mask = attention_mask.reshape(1, 1)
            elif attention_mask.ndim == 1:
                attention_mask = attention_mask.reshape(1, -1)
        else:
            attention_mask = np.ones_like(input_ids)

        if input_ids.shape[1] < length:
            raw_pad = getattr(hf, "pad_token_id", 0)
            pad_id = int(raw_pad) if isinstance(raw_pad, (int, np.integer)) else 0
            pad_width = length - input_ids.shape[1]
            input_ids = np.pad(input_ids, ((0, 0), (0, pad_width)), mode="constant", constant_values=pad_id)
            attention_mask = np.pad(attention_mask, ((0, 0), (0, pad_width)), mode="constant", constant_values=0)

        return mx.array(input_ids), mx.array(attention_mask)

    @staticmethod
    def positions(attention_mask: mx.array) -> mx.array:
        # cumsum(mask) - 1, padded positions set to 1
        positions = mx.cumsum(attention_mask.astype(mx.int32), axis=-1) - 1
        return mx.where(attention_mask == 0, 1, positions)

    @staticmethod
    def encode_prompt(
        prompt: str,
        tokenizer: Tokenizer,
        text_encoder: Qwen3TextEncoder,
    ) -> tuple[mx.array, mx.array]:
        input_ids, attention_mask = Flux3PromptEncoder.tokenize(prompt, tokenizer)
        tokens = text_encoder.get_prompt_embeds(
            input_ids=input_ids,
            attention_mask=attention_mask,
            hidden_state_layers=Flux3PromptEncoder.OUTPUT_LAYERS,
            position_ids=Flux3PromptEncoder.positions(attention_mask),
        )
        return tokens.astype(mx.bfloat16), Flux3PromptEncoder.prepare_text_ids(tokens.shape[1])

    @staticmethod
    def prepare_text_ids(length: int) -> mx.array:
        # (0, 0, 0, position) with position = arange(L); float32 as the transformer expects.
        length = max(1, length)
        ids = np.zeros((length, 4), dtype=np.float32)
        ids[:, 3] = np.arange(length)
        return mx.array(ids)