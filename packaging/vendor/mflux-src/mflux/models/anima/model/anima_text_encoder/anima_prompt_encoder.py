import mlx.core as mx

from mflux.models.anima.model.anima_text_encoder.anima_prompt_weights import AnimaPromptWeights
from mflux.models.anima.model.anima_transformer.anima_transformer import AnimaTransformer
from mflux.models.common.tokenizer.tokenizer import BaseTokenizer
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder


class AnimaPromptEncoder:
    # The reference reads an empty prompt as this one Qwen3 token (<|endoftext|>).
    QWEN_PAD_TOKEN = 151643

    @staticmethod
    def encode(
        prompt: str,
        qwen_tokenizer: BaseTokenizer,
        t5_tokenizer: BaseTokenizer,
        text_encoder: Qwen3TextEncoder,
        transformer: AnimaTransformer,
        weighted: bool = True,
    ) -> mx.array:
        # The prompt goes in twice: Qwen3 token ids for the encoder (no special tokens), and T5 token ids (with the
        # closing </s>) for the adapter that turns the Qwen3 states into the transformer's context.
        # weighted=False reads ComfyUI's (text:1.2) syntax as plain text, as training captions are. Their escaped
        # parentheses still become literal ones, so a caption trains on the text a prompt with the same tags gives
        parts = AnimaPromptWeights.parse(prompt) if weighted else [(AnimaPromptWeights.unescape(prompt), 1.0)]
        if AnimaPromptWeights.is_plain(parts):
            text = parts[0][0] if parts else ""
            qwen_ids = qwen_tokenizer.tokenize(text, keep_empty=True).input_ids
            t5_ids = t5_tokenizer.tokenize(text, keep_empty=True).input_ids
            t5_weights = None
        else:
            qwen_ids, t5_ids, t5_weights = AnimaPromptEncoder._weighted_ids(parts, qwen_tokenizer, t5_tokenizer)
        if qwen_ids.size == 0:
            qwen_ids = mx.array([[AnimaPromptEncoder.QWEN_PAD_TOKEN]])
        # The text path runs once per prompt, in float32 like the reference's text encoder. In bfloat16 the first
        # denoising steps drift twice as far from the float32 result, most on short prompts.
        hidden_states, _ = text_encoder(qwen_ids, dtype=mx.float32)
        return transformer.encode_text(hidden_states, t5_ids, t5_weights)

    @staticmethod
    def _weighted_ids(
        parts: list[tuple[str, float]], qwen_tokenizer: BaseTokenizer, t5_tokenizer: BaseTokenizer
    ) -> tuple[mx.array, mx.array, mx.array]:
        # Each part tokenized on its own, as ComfyUI does, then joined; T5 drops each part's </s> and ends with one
        qwen, t5, weights = [], [], []
        for text, weight in parts:
            qwen += qwen_tokenizer.tokenizer(text, add_special_tokens=False)["input_ids"]
            ids = t5_tokenizer.tokenizer(text)["input_ids"][:-1]
            t5 += ids
            weights += [weight] * len(ids)
        # Cut at the tokenizers' length like the plain path, keeping the closing </s>
        limit = qwen_tokenizer.max_length
        t5 = t5[: t5_tokenizer.max_length - 1] + [t5_tokenizer.tokenizer.eos_token_id]
        weights = weights[: t5_tokenizer.max_length - 1] + [1.0]
        return mx.array([qwen[:limit]]), mx.array([t5]), mx.array(weights)
