from __future__ import annotations

import argparse
from pathlib import Path

from gen26.chunking import TokenBudget, format_budget_report
from gen26.latex_parser import load_latex_source, parse_loaded_source
from gen26.run_store import RunStore, apply_node_states
from gen26.terminal_planner import terminal_plan


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

    resume_parser = subparsers.add_parser(
        "resume",
        help="Resume a previous digestion run from its Markdown output path.",
    )
    resume_parser.add_argument("output", type=Path, help="Existing Markdown output path.")
    return parser


def add_digest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("digestion.md"),
        help="Markdown file that receives streamed chunk outputs and final abstract.",
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


class RuntimeTokenCounter:
    name = "gemma3"

    def __init__(self, runtime) -> None:
        self.runtime = runtime

    def count(self, text: str) -> int:
        return self.runtime.count_tokens(text)


def run_digest(args) -> int:
    from gen26.digestion import digest_chunks
    from gen26.gemma_runtime import GemmaDigestRuntime

    runtime = GemmaDigestRuntime()
    source = load_latex_source(args.source)
    try:
        root = parse_loaded_source(source, RuntimeTokenCounter(runtime))
        budget = TokenBudget(
            cache_length=runtime.cache_length,
            usable_input_tokens=runtime.safe_input_tokens,
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

    runtime = GemmaDigestRuntime()
    source_path = Path(state["source"])
    source = load_latex_source(source_path)
    try:
        root = parse_loaded_source(source, RuntimeTokenCounter(runtime))
        apply_node_states(root, state.get("node_states", []))
        budget = TokenBudget(
            cache_length=runtime.cache_length,
            usable_input_tokens=runtime.safe_input_tokens,
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
