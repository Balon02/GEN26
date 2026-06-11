import argparse
import gc
import os
import time
from pathlib import Path

import kagglehub
from dotenv import load_dotenv


os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "1.0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "vmm")

import jax
from jax import numpy as jnp

from gemma import gm, peft


load_dotenv("/home/balon/source/GEN26/.env")

REPO_ROOT = Path(__file__).resolve().parent
GEMMA_PATH = kagglehub.model_download("google/gemma-3/flax/gemma3-4b-it")
CKPT_PATH = os.path.join(GEMMA_PATH, "gemma3-4b-it")
TOKENIZER_PATH = os.path.join(GEMMA_PATH, "tokenizer.model")
DEFAULT_INT8_CHECKPOINT = (
    REPO_ROOT / ".cache" / "gemma" / "checkpoints" / "gemma3-4b-it-text-int8"
)
DEFAULT_INT4_CHECKPOINT = (
    REPO_ROOT / ".cache" / "gemma" / "checkpoints" / "gemma3-4b-it-text-int4"
)

PROMPT_UNIT = r"""
\section{Background}
We study a transformer-based digestion chain for scientific LaTeX papers.
The system segments source files by document structure, preserves equation and
figure references, summarizes sections in order, and carries forward a compact
running abstract. Important entities include hypotheses, datasets, methods,
metrics, ablations, limitations, and figure captions. The answer should remain
faithful to the provided source and avoid inventing claims.
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe usable Gemma 3 context length by doubling cache_length."
    )
    parser.add_argument("--start-cache", type=int, default=2048)
    parser.add_argument("--max-cache", type=int, default=32768)
    parser.add_argument(
        "--step-cache",
        type=int,
        default=None,
        help="Cache-length increment. Defaults to doubling when omitted.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--fill-ratio", type=float, default=0.90)
    parser.add_argument("--safety-margin", type=int, default=32)
    parser.add_argument("--multi-turn", action="store_true")
    parser.add_argument("--print-response", action="store_true")
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Load only text params for an unquantized baseline.",
    )
    parser.add_argument(
        "--quantization",
        choices=("none", "int4", "int8", "int4-saved", "int8-saved"),
        default="none",
        help=(
            "Use the official peft.quantize sampling path. Quantized modes are "
            "text-only. 'int4'/'int8' quantize in this process; '*-saved' loads "
            "a previously saved quantized checkpoint."
        ),
    )
    parser.add_argument(
        "--int8-checkpoint",
        type=Path,
        default=DEFAULT_INT8_CHECKPOINT,
        help="Saved INT8 params checkpoint used by --quantization int8-saved.",
    )
    parser.add_argument(
        "--int4-checkpoint",
        type=Path,
        default=DEFAULT_INT4_CHECKPOINT,
        help="Saved INT4 params checkpoint used by --quantization int4-saved.",
    )
    parser.add_argument(
        "--quantized-scope",
        choices=("text", "full"),
        default="text",
        help=(
            "Scope for quantized modes. 'full' loads/preserves vision params, "
            "but peft.quantize only quantizes recognized text transformer "
            "weights."
        ),
    )
    return parser.parse_args()


def make_instruction(body: str) -> str:
    return (
        "<start_of_turn>user\n"
        "Summarize the following synthetic LaTeX-like research-paper content "
        "in one sentence. Focus on the method and limitations.\n\n"
        f"{body}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def count_tokens(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_bos=True))


def build_prompt_near_token_budget(tokenizer, target_tokens: int) -> tuple[str, int]:
    if target_tokens < 64:
        raise ValueError(f"target_tokens is too small: {target_tokens}")

    low = 0
    high = 1
    while count_tokens(tokenizer, make_instruction(PROMPT_UNIT * high)) <= target_tokens:
        low = high
        high *= 2

    while low + 1 < high:
        mid = (low + high) // 2
        prompt = make_instruction(PROMPT_UNIT * mid)
        if count_tokens(tokenizer, prompt) <= target_tokens:
            low = mid
        else:
            high = mid

    prompt = make_instruction(PROMPT_UNIT * low)
    return prompt, count_tokens(tokenizer, prompt)


def iter_cache_lengths(start: int, max_cache: int, step: int | None = None):
    cache_length = start
    while cache_length <= max_cache:
        yield cache_length
        if step is None:
            cache_length *= 2
        else:
            cache_length += step


def load_model_and_params(args):
    if args.quantization in {"int4-saved", "int8-saved"}:
        is_int4 = args.quantization == "int4-saved"
        checkpoint = args.int4_checkpoint if is_int4 else args.int8_checkpoint
        if args.quantized_scope == "full":
            default_checkpoint = (
                DEFAULT_INT4_CHECKPOINT if is_int4 else DEFAULT_INT8_CHECKPOINT
            )
            if checkpoint == default_checkpoint:
                checkpoint = checkpoint.with_name(
                    checkpoint.name.replace("-text-", "-full-")
                )
        checkpoint = checkpoint.expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Saved checkpoint does not exist: {checkpoint}\n"
                "Create it first with: uv run python prequantize_gemma3_int8.py "
                f"--method {'int4' if is_int4 else 'int8'} "
                f"--scope {args.quantized_scope}"
            )

        dtype = jnp.int4 if is_int4 else jnp.int8
        text_only = args.quantized_scope == "text"
        print(f"INIT quantization={args.quantization} text_only={text_only}")
        model = gm.nn.IntWrapper(
            model=gm.nn.Gemma3_4B(text_only=text_only, dtype=dtype),
            dtype=dtype,
        )

        print(f"MODEL_LOAD saved params from {checkpoint}")
        params = gm.ckpts.load_params(checkpoint, text_only=text_only)
        return model, params

    if args.quantization in {"int4", "int8"}:
        is_int4 = args.quantization == "int4"
        dtype = jnp.int4 if is_int4 else jnp.int8
        method = peft.QuantizationMethod.INT4 if is_int4 else peft.QuantizationMethod.INT8
        text_only = args.quantized_scope == "text"
        print(f"INIT quantization={args.quantization} text_only={text_only}")
        model = gm.nn.IntWrapper(
            model=gm.nn.Gemma3_4B(text_only=text_only, dtype=dtype),
            dtype=dtype,
        )

        print(f"MODEL_LOAD raw params scope={args.quantized_scope}")
        original = gm.ckpts.load_params(CKPT_PATH, text_only=text_only)

        print(f"MODEL_QUANTIZE peft.quantize {args.quantization.upper()}")
        params = peft.quantize(
            original,
            method=method,
            checkpoint_kernel_key="w",
        )
        del original
        gc.collect()
        jax.clear_caches()
        return model, params

    print(f"INIT quantization=none text_only={args.text_only}")
    model = gm.nn.Gemma3_4B(text_only=args.text_only)

    print("MODEL_LOAD")
    params = gm.ckpts.load_params(CKPT_PATH, text_only=args.text_only)
    return model, params


def main():
    args = parse_args()

    model, params = load_model_and_params(args)

    print("TOKENIZER_LOAD")
    tokenizer = gm.text.Gemma3Tokenizer(TOKENIZER_PATH)

    last_state = None
    for cache_length in iter_cache_lengths(
        args.start_cache,
        args.max_cache,
        args.step_cache,
    ):
        target_tokens = int(cache_length * args.fill_ratio)
        target_tokens -= args.max_new_tokens + args.safety_margin
        prompt, prompt_tokens = build_prompt_near_token_budget(tokenizer, target_tokens)

        print(
            "\n"
            f"cache_length={cache_length} "
            f"target_tokens={target_tokens} "
            f"prompt_tokens={prompt_tokens} "
            f"max_new_tokens={args.max_new_tokens}",
            flush=True,
        )

        sampler = gm.text.Sampler(
            model=model,
            params=params,
            tokenizer=tokenizer,
            cache_length=cache_length,
            max_out_length=args.max_new_tokens,
            pad_length=None,
        )

        start = time.perf_counter()
        try:
            output = sampler.sample(
                prompt,
                max_new_tokens=args.max_new_tokens,
                return_state=True,
                last_state=last_state if args.multi_turn else None,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            print(
                f"FAIL cache_length={cache_length} "
                f"prompt_tokens={prompt_tokens} "
                f"elapsed={elapsed:.1f}s "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )
            raise

        elapsed = time.perf_counter() - start
        text = output.text if hasattr(output, "text") else str(output)
        if args.multi_turn:
            last_state = output.state

        print(
            f"OK cache_length={cache_length} "
            f"prompt_tokens={prompt_tokens} "
            f"elapsed={elapsed:.1f}s "
            f"response_chars={len(text)}",
            flush=True,
        )
        if args.print_response:
            print(text, flush=True)


if __name__ == "__main__":
    main()
