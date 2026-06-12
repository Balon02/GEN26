from __future__ import annotations

import argparse
from pathlib import Path

from gen26.chunking import format_budget_report, make_token_budget
from gen26.latex_parser import load_latex_source, parse_loaded_source
from gen26.run_store import RunStore, apply_node_states
from gen26.terminal_planner import terminal_plan


DEFAULT_MAX_TOKENS = 10240
DEFAULT_CONTEXT_SCALE = 1.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gen26",
        description="Token-aware Gemma digestion for LaTeX research papers.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    digest_parser = subparsers.add_parser(
        "digest",
        help="Run sequential Gemma digestion with the curses planner.",
    )
    digest_parser.add_argument("source", type=Path, help=".tex file, directory, or .tar.gz")
    add_digest_args(digest_parser)
    add_runtime_args(digest_parser)

    resume_parser = subparsers.add_parser(
        "resume",
        help="Resume a previous digestion run from its Markdown output path.",
    )
    resume_parser.add_argument("output", type=Path, help="Existing Markdown output path.")
    add_runtime_args(resume_parser)
    return parser


def add_digest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("digestion.md"),
        help="Markdown file that receives streamed chunk outputs and final abstract.",
    )


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Gemma sampler cache length. Lower this on smaller GPUs or raise "
            "it on larger accelerators; usable input context is derived from "
            "it. Defaults to 10240 for new runs and to the stored cache "
            "length when resuming."
        ),
    )
    parser.add_argument(
        "--context-scale",
        type=float,
        default=None,
        help=(
            "Multiplier for fixed context-token allocations such as rolling "
            "memory, instruction reservation, memory deltas, image notes, and "
            "final prompt fitting limits. Defaults to 1.0 for new runs and to "
            "the stored value when resuming."
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "digest":
        return run_digest(args)
    if args.command == "resume":
        return run_resume(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


def max_tokens_from_args(args, state: dict | None = None) -> int:
    if args.max_tokens is not None:
        return args.max_tokens
    if state is not None:
        stored = state.get("runtime", {}).get("cache_length")
        if stored is not None:
            return int(stored)
    return DEFAULT_MAX_TOKENS


def context_scale_from_args(args, state: dict | None = None) -> float:
    if args.context_scale is not None:
        if args.context_scale <= 0:
            raise ValueError("--context-scale must be greater than zero.")
        return args.context_scale
    if state is not None:
        stored = state.get("budget", {}).get("context_scale")
        if stored is not None:
            return float(stored)
    return DEFAULT_CONTEXT_SCALE


class RuntimeTokenCounter:
    name = "gemma3"

    def __init__(self, runtime) -> None:
        self.runtime = runtime

    def count(self, text: str) -> int:
        return self.runtime.count_tokens(text)


def run_digest(args) -> int:
    from gen26.digestion import digest_chunks
    from gen26.gemma_runtime import GemmaDigestRuntime

    runtime = GemmaDigestRuntime(max_tokens=max_tokens_from_args(args))
    context_scale = context_scale_from_args(args)
    source = load_latex_source(args.source)
    try:
        root = parse_loaded_source(source, RuntimeTokenCounter(runtime))
        budget = make_token_budget(
            cache_length=runtime.cache_length,
            usable_input_tokens=runtime.safe_input_tokens,
            context_scale=context_scale,
        )
        chunks = terminal_plan(root, budget)
        print(format_budget_report(chunks, budget), flush=True)
        store = RunStore(args.output)
        store.create(args.source.resolve(), runtime, budget, root, chunks)
        digest_chunks(
            runtime,
            chunks,
            output_file=args.output,
            rolling_memory_token_limit=budget.rolling_memory_tokens,
            context_scale=context_scale,
            run_store=store,
        )
        print(f"\nMarkdown output: {args.output}", flush=True)
        return 0
    finally:
        source.cleanup()


def run_resume(args) -> int:
    from gen26.digestion import RollingMemory, digest_chunks
    from gen26.gemma_runtime import GemmaDigestRuntime

    store = RunStore(args.output)
    state = store.load()
    store.mark_interrupted_chunks()

    runtime = GemmaDigestRuntime(max_tokens=max_tokens_from_args(args, state))
    context_scale = context_scale_from_args(args, state)
    source_path = Path(state["source"])
    source = load_latex_source(source_path)
    try:
        root = parse_loaded_source(source, RuntimeTokenCounter(runtime))
        apply_node_states(root, state.get("node_states", []))
        budget = make_token_budget(
            cache_length=runtime.cache_length,
            usable_input_tokens=runtime.safe_input_tokens,
            context_scale=context_scale,
        )
        chunks = terminal_plan(root, budget)
        prefix = store.update_plan(root, chunks)
        print(format_budget_report(chunks, budget), flush=True)
        print(f"Resuming at chunk {prefix + 1}.", flush=True)

        memory = RollingMemory(text=store.state.get("rolling_memory") or RollingMemory().text)
        summaries = store.state.get("completed_summaries", [])[:prefix]
        digest_chunks(
            runtime,
            chunks[prefix:],
            output_file=args.output,
            rolling_memory_token_limit=budget.rolling_memory_tokens,
            context_scale=context_scale,
            run_store=store,
            initial_memory=memory,
            initial_summaries=summaries,
            append_output=True,
            total_chunks=len(chunks),
        )
        print(f"\nMarkdown output: {args.output}", flush=True)
        return 0
    finally:
        source.cleanup()
