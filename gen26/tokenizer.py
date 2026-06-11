from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TokenCounter:
    name: str

    def count(self, text: str) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class ApproxTokenCounter(TokenCounter):
    name: str = "approx"

    def count(self, text: str) -> int:
        if not text.strip():
            return 0
        # Conservative enough for CLI previews when Gemma is unavailable.
        return max(1, round(len(text) / 3.6))


class Gemma3TokenCounter(TokenCounter):
    def __init__(self, tokenizer_path: Path | None = None) -> None:
        from gemma import gm
        import kagglehub

        if tokenizer_path is None:
            gemma_path = kagglehub.model_download("google/gemma-3/flax/gemma3-4b-it")
            tokenizer_path = Path(gemma_path) / "tokenizer.model"
        self.name = "gemma3"
        self._tokenizer = gm.text.Gemma3Tokenizer(os.fspath(tokenizer_path))

    def count(self, text: str) -> int:
        if not text.strip():
            return 0
        return len(self._tokenizer.encode(text, add_bos=True))


def load_token_counter(kind: str, tokenizer_path: Path | None = None) -> TokenCounter:
    if kind == "gemma3":
        return Gemma3TokenCounter(tokenizer_path)
    if kind == "approx":
        return ApproxTokenCounter()
    raise ValueError(f"Unknown tokenizer: {kind}")

