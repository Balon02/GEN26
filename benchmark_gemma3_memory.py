import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


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
        description="Benchmark Gemma 3 VRAM by model variant and cache length."
    )
    parser.add_argument(
        "--variants",
        default="original,int8,int4",
        help="Comma-separated variants: original,int8,int4.",
    )
    parser.add_argument("--scope", choices=("text", "full"), default="text")
    parser.add_argument("--start-cache", type=int, default=2048)
    parser.add_argument("--max-cache", type=int, default=16384)
    parser.add_argument(
        "--step-cache",
        type=int,
        default=None,
        help="Cache-length increment. Defaults to --start-cache.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--fill-ratio", type=float, default=0.90)
    parser.add_argument("--safety-margin", type=int, default=32)
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="Only measure import/model/load/quantization phases.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("benchmark_gemma3_memory.log"),
        help="Append benchmark output to this file.",
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--variant", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def default_args():
    return argparse.Namespace(
        variants="original,int8,int4",
        scope="text",
        start_cache=2048,
        max_cache=16384,
        step_cache=2048,
        max_new_tokens=16,
        fill_ratio=0.90,
        safety_margin=32,
        load_only=False,
        log_file=Path("benchmark_gemma3_memory.log"),
        child=False,
        variant=None,
    )


def emit(message):
    log_file = os.environ.get("GEMMA3_MEMORY_BENCH_LOG")
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message)
            f.write("\n")
    else:
        print(message, flush=True)


def gpu_memory():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"
    used, free = result.stdout.strip().splitlines()[0].split(",")
    return int(used.strip()), int(free.strip()), None


def run_parent(args):
    log_file = args.log_file.expanduser().resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(
            f"benchmark_gemma3_memory log started_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(
            f"variants={args.variants} scope={args.scope} "
            f"start_cache={args.start_cache} max_cache={args.max_cache} "
            f"step_cache={args.step_cache or args.start_cache} "
            f"max_new_tokens={args.max_new_tokens} stream=True "
            f"load_only={args.load_only}\n"
        )
    print(f"log_file={log_file}", flush=True)

    variants = [part.strip() for part in args.variants.split(",") if part.strip()]
    step_cache = args.step_cache or args.start_cache
    for variant in variants:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n===== variant={variant} scope={args.scope} =====\n")
        cmd = [
            sys.executable,
            __file__,
            "--child",
            "--variant",
            variant,
            "--scope",
            args.scope,
            "--start-cache",
            str(args.start_cache),
            "--max-cache",
            str(args.max_cache),
            "--step-cache",
            str(step_cache),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--fill-ratio",
            str(args.fill_ratio),
            "--safety-margin",
            str(args.safety_margin),
        ]
        if args.load_only:
            cmd.append("--load-only")

        child_env = os.environ.copy()
        child_env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"
        child_env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        child_env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "vmm"
        child_env["GEMMA3_MEMORY_BENCH_LOG"] = str(log_file)
        with open(log_file, "a", encoding="utf-8") as f:
            result = subprocess.run(
                cmd,
                env=child_env,
                check=False,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
            )
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"child_exit variant={variant} returncode={result.returncode}\n")


def cache_lengths(start, maximum, step):
    value = start
    while value <= maximum:
        yield value
        value += step


def make_instruction(body):
    return (
        "<start_of_turn>user\n"
        "Summarize the following synthetic LaTeX-like research-paper content "
        "in one sentence. Focus on the method and limitations.\n\n"
        f"{body}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def count_tokens(tokenizer, prompt):
    return len(tokenizer.encode(prompt, add_bos=True))


def build_prompt_near_token_budget(tokenizer, target_tokens):
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


def log_memory(label, baseline, started_at, **fields):
    used, free, error = gpu_memory()
    elapsed = time.perf_counter() - started_at
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    if error:
        emit(
            f"MEM {label} used_mb=NA free_mb=NA delta_mb=NA "
            f"elapsed_s={elapsed:.1f} nvidia_smi_error={error} {details}"
        )
        return
    emit(
        f"MEM {label} used_mb={used} free_mb={free} "
        f"delta_mb={used - baseline} elapsed_s={elapsed:.1f} {details}",
    )


def run_child(args):
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "vmm"

    started_at = time.perf_counter()
    baseline, _, error = gpu_memory()
    if error:
        baseline = 0
    log_memory("process_start", baseline, started_at)

    import gc

    import kagglehub
    from dotenv import load_dotenv

    import jax
    from jax import numpy as jnp

    from gemma import gm, peft

    load_dotenv("/home/balon/source/GEN26/.env")

    gemma_path = kagglehub.model_download("google/gemma-3/flax/gemma3-4b-it")
    ckpt_path = os.path.join(gemma_path, "gemma3-4b-it")
    tokenizer_path = os.path.join(gemma_path, "tokenizer.model")
    text_only = args.scope == "text"

    log_memory("after_imports", baseline, started_at)

    if args.variant == "original":
        model = gm.nn.Gemma3_4B(text_only=text_only)
    elif args.variant in {"int8", "int4"}:
        dtype = jnp.int8 if args.variant == "int8" else jnp.int4
        model = gm.nn.IntWrapper(
            model=gm.nn.Gemma3_4B(text_only=text_only),
            dtype=dtype,
        )
    else:
        raise ValueError(f"Unknown variant: {args.variant}")

    log_memory("after_model_init", baseline, started_at)

    params = gm.ckpts.load_params(ckpt_path, text_only=text_only)
    log_memory("after_params_load", baseline, started_at)

    if args.variant in {"int8", "int4"}:
        method = (
            peft.QuantizationMethod.INT8
            if args.variant == "int8"
            else peft.QuantizationMethod.INT4
        )
        params = peft.quantize(
            params,
            method=method,
            checkpoint_kernel_key="w",
        )
        params = jax.block_until_ready(params)
        gc.collect()
        jax.clear_caches()
        log_memory("after_quantize", baseline, started_at)

    tokenizer = gm.text.Gemma3Tokenizer(tokenizer_path)
    log_memory("after_tokenizer", baseline, started_at)

    if args.load_only:
        return

    step_cache = args.step_cache or args.start_cache
    for cache_length in cache_lengths(args.start_cache, args.max_cache, step_cache):
        target_tokens = int(cache_length * args.fill_ratio)
        target_tokens -= args.max_new_tokens + args.safety_margin
        prompt, prompt_tokens = build_prompt_near_token_budget(tokenizer, target_tokens)

        sampler = gm.text.Sampler(
            model=model,
            params=params,
            tokenizer=tokenizer,
            cache_length=cache_length,
            max_out_length=args.max_new_tokens,
            pad_length=None,
        )
        log_memory(
            "before_sample",
            baseline,
            started_at,
            cache_length=cache_length,
            prompt_tokens=prompt_tokens,
        )
        try:
            response_chars = 0
            streamed_chunks = 0
            stream = sampler.sample(
                prompt,
                max_new_tokens=args.max_new_tokens,
                stream=True,
                return_state=False,
            )
            for chunk in stream:
                streamed_chunks += 1
                response_chars += len(chunk)
                if streamed_chunks == 1:
                    log_memory(
                        "after_stream_first",
                        baseline,
                        started_at,
                        cache_length=cache_length,
                        prompt_tokens=prompt_tokens,
                    )
            log_memory(
                "after_sample_ok",
                baseline,
                started_at,
                cache_length=cache_length,
                prompt_tokens=prompt_tokens,
                streamed_chunks=streamed_chunks,
                response_chars=response_chars,
            )
        except Exception as exc:
            log_memory(
                "after_sample_fail",
                baseline,
                started_at,
                cache_length=cache_length,
                prompt_tokens=prompt_tokens,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise


def main():
    if len(sys.argv) == 1:
        args = default_args()
    else:
        args = parse_args()
    if args.child:
        run_child(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
