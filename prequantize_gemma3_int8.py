import argparse
import gc
import os
import shutil
from pathlib import Path

import kagglehub
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parent

os.environ["JAX_PLATFORMS"] = "cpu"

# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
# # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "vmm"
# # os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

import jax
import orbax.checkpoint as ocp
from gemma import gm, peft


load_dotenv("/home/balon/source/GEN26/.env")

GEMMA_PATH = kagglehub.model_download("google/gemma-3/flax/gemma3-4b-it")
CKPT_PATH = os.path.join(GEMMA_PATH, "gemma3-4b-it")
DEFAULT_OUTPUT_DIR = REPO_ROOT / ".cache" / "gemma" / "checkpoints"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a saved quantized Gemma 3 4B params checkpoint using "
            "the official gm.ckpts + peft.quantize path."
        )
    )
    parser.add_argument(
        "--method",
        choices=("int4", "int8"),
        default="int8",
        help="Quantization method passed to peft.quantize.",
    )
    parser.add_argument(
        "--scope",
        choices=("text", "full"),
        default="text",
        help=(
            "Checkpoint scope to load before quantization. 'full' preserves "
            "vision params but peft.quantize only quantizes recognized text "
            "transformer weights."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Orbax checkpoint directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output checkpoint before saving.",
    )
    return parser.parse_args()


def tree_bytes_by_dtype(tree):
    by_dtype = {}
    for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "dtype") and hasattr(leaf, "size"):
            by_dtype[str(leaf.dtype)] = by_dtype.get(str(leaf.dtype), 0) + (
                leaf.size * leaf.dtype.itemsize
            )
    return by_dtype


def print_tree_summary(label, tree):
    by_dtype = tree_bytes_by_dtype(tree)
    total = sum(by_dtype.values())
    gib = 1024**3
    print(f"{label}_total_gib={total / gib:.2f}", flush=True)
    print(
        f"{label}_by_dtype_gib="
        f"{ {dtype: round(size / gib, 3) for dtype, size in by_dtype.items()} }",
        flush=True,
    )


def main():
    args = parse_args()
    text_only = args.scope == "text"
    method = peft.QuantizationMethod(args.method)
    output = (
        args.output
        or DEFAULT_OUTPUT_DIR / f"gemma3-4b-it-{args.scope}-{args.method}"
    )
    output = output.expanduser().resolve()

    if output.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Output checkpoint already exists: {output}\n"
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(output)
    tmp_output = output.with_name(f"{output.name}.orbax-checkpoint-tmp")
    if tmp_output.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Temporary checkpoint directory already exists: {tmp_output}\n"
                "Pass --overwrite to remove it."
            )
        shutil.rmtree(tmp_output)

    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"source_checkpoint={CKPT_PATH}", flush=True)
    print(f"output_checkpoint={output}", flush=True)

    print(f"loading raw params scope={args.scope}...", flush=True)
    raw_params = gm.ckpts.load_params(CKPT_PATH, text_only=text_only)
    print_tree_summary(f"raw_{args.scope}", raw_params)

    print(f"quantizing params to {args.method.upper()}...", flush=True)
    quantized_params = peft.quantize(
        raw_params,
        method=method,
        checkpoint_kernel_key="w",
    )
    del raw_params
    gc.collect()
    jax.clear_caches()
    print(f"materializing {args.method.upper()} params before save...", flush=True)
    quantized_params = jax.block_until_ready(quantized_params)
    print_tree_summary(f"quantized_{args.scope}_{args.method}", quantized_params)

    print(f"saving {args.method.upper()} params checkpoint...", flush=True)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(output, quantized_params, force=True)
    checkpointer.wait_until_finished()
    print("done", flush=True)


if __name__ == "__main__":
    main()
