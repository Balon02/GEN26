from __future__ import annotations

import os
import sys
from pathlib import Path


# Match smoke_test_g3.py before importing Gemma/JAX.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "1.0")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "vmm")

import kagglehub  # noqa: E402
import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from gemma import gm  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
GEMMA_MODEL = "google/gemma-3/flax/gemma3-4b-it"
DEFAULT_CACHE_LENGTH = 10240
DEFAULT_SAFE_INPUT_TOKENS = 7800
CACHE_TO_INPUT_RESERVE = DEFAULT_CACHE_LENGTH - DEFAULT_SAFE_INPUT_TOKENS
MAX_OUTPUT_TOKENS = 768
FINAL_OUTPUT_TOKENS = 3072


def safe_input_tokens_for_cache(cache_length: int) -> int:
    """Derive usable prompt budget from the single runtime cache-size knob."""

    safe_input_tokens = min(
        DEFAULT_SAFE_INPUT_TOKENS,
        cache_length - CACHE_TO_INPUT_RESERVE,
    )
    if safe_input_tokens <= 0:
        raise ValueError(
            f"max_tokens={cache_length} is too small; it leaves no usable "
            "input context after the fixed cache reserve."
        )
    return safe_input_tokens


class GemmaDigestRuntime:
    def __init__(self, max_tokens: int = DEFAULT_CACHE_LENGTH) -> None:
        load_dotenv(str(REPO_ROOT / ".env"))

        print("INIT", flush=True)
        gemma_path = kagglehub.model_download(GEMMA_MODEL)
        ckpt_path = os.path.join(gemma_path, "gemma3-4b-it")
        tokenizer_path = os.path.join(gemma_path, "tokenizer.model")

        self.model = gm.nn.Gemma3_4B(text_only=False)
        print("MODEL_LOAD", flush=True)
        self.params = gm.ckpts.load_params(ckpt_path)
        print("TOKENIZER_LOAD", flush=True)
        self.tokenizer = gm.text.Gemma3Tokenizer(tokenizer_path)
        self.cache_length = max_tokens
        self.safe_input_tokens = safe_input_tokens_for_cache(max_tokens)
        self.max_output_tokens = MAX_OUTPUT_TOKENS
        self.final_output_tokens = FINAL_OUTPUT_TOKENS
        self.image_height = self.model.config.vision_encoder.image_height
        self.image_width = self.model.config.vision_encoder.image_width
        if self.image_height != self.image_width:
            raise ValueError("Expected square Gemma image input.")
        self.image_size = self.image_height

        print("SAMPLER_SETUP", flush=True)
        self.sampler = gm.text.Sampler(
            model=self.model,
            params=self.params,
            tokenizer=self.tokenizer,
            cache_length=self.cache_length,
            max_out_length=self.max_output_tokens,
        )
        self.final_sampler = gm.text.Sampler(
            model=self.model,
            params=self.params,
            tokenizer=self.tokenizer,
            cache_length=self.cache_length,
            max_out_length=self.final_output_tokens,
        )

    def format_prompt(self, prompt: str) -> str:
        prompt = prompt.replace("<|image|>", "<start_of_image>")
        return (
            "<start_of_turn>user\n"
            f"{prompt}"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

    def count_tokens(self, text: str) -> int:
        if not text.strip():
            return 0
        return len(self.tokenizer.encode(text, add_bos=True))

    def count_prompt_tokens(self, prompt: str) -> int:
        return len(self.tokenizer.encode(self.format_prompt(prompt), add_bos=True))

    def chat(
        self,
        prompt: str,
        images: list[object] | None = None,
        stream_file=None,
        max_new_tokens: int | None = None,
    ) -> str:
        formatted = self.format_prompt(prompt)
        sampler_images = normalize_images_for_sampler(images)
        image_count = count_sampler_images(sampler_images)
        validate_image_placeholders(formatted, image_count)
        tokens: list[str] = []
        output_tokens = max_new_tokens or self.max_output_tokens
        sampler = self.final_sampler if output_tokens > self.max_output_tokens else self.sampler
        stream = sampler.sample(
            formatted,
            images=sampler_images,
            max_new_tokens=output_tokens,
            stream=True,
        )
        for token in stream:
            text = token.text if hasattr(token, "text") else str(token)
            if text in {"<end_of_turn>", "<eos>"}:
                continue
            tokens.append(text)
            sys.stdout.write(text)
            sys.stdout.flush()
            if stream_file is not None:
                stream_file.write(text)
                stream_file.flush()
        return "".join(tokens).strip()


def normalize_images_for_sampler(images: list[object] | object | None):
    """Return image shape expected by Sampler for one unbatched prompt.

    `gm.text.Sampler.sample()` receives a plain string prompt here. Its internal
    normalizer adds the batch dimension, so we must pass H,W,C or N,H,W,C.
    Passing B,N,H,W,C would become 6D inside Gemma.
    """
    if images is None:
        return None
    array = np.asarray(images, dtype=np.uint8)
    if array.ndim not in {3, 4}:
        raise ValueError(
            "Images must have shape H,W,C or N,H,W,C for one prompt; "
            f"got {array.shape}."
        )
    if array.shape[-1] != 3:
        raise ValueError(f"Expected RGB images with 3 channels; got {array.shape}.")
    return array


def count_sampler_images(images) -> int:
    if images is None:
        return 0
    if images.ndim == 3:
        return 1
    if images.ndim == 4:
        return images.shape[0]
    raise ValueError(f"Unexpected sampler image shape: {images.shape}.")


def validate_image_placeholders(prompt: str, image_count: int) -> None:
    placeholders = prompt.count("<start_of_image>")
    if placeholders != image_count:
        raise ValueError(
            "Prompt/image mismatch: "
            f"{placeholders} image placeholders for {image_count} images."
        )
