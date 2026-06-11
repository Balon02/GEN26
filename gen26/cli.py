from __future__ import annotations

import argparse
from pathlib import Path

from gen26.chunking import (
    TokenBudget,
    apply_selection_edits,
    format_budget_report,
    pack_chunks,
)
from gen26.latex_parser import load_latex_source, parse_latex_project, parse_loaded_source
from gen26.paper_tree import print_tree
from gen26.planner import interactive_plan
from gen26.run_store import RunStore, apply_node_states
from gen26.terminal_planner import terminal_plan
from gen26.tokenizer import load_token_counter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gen26",
        description="Token-aware CLI tools for digesting LaTeX research papers.",
    )
    parser.add_argument(
        "--tokenizer",
        choices=("gemma3", "approx"),
        default="gemma3",
        help="Tokenizer used for node counts. Use approx only for local parser checks.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Optional path to tokenizer.model for --tokenizer gemma3.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("ingest", "Parse a LaTeX project and print a compact summary."),
        ("tree", "Parse a LaTeX project and print the token-aware section tree."),
        ("budget", "Parse, pack selected nodes, and print a chunk budget report."),
        ("digest", "Run sequential Gemma digestion with rolling memory."),
    ):
        subparser = subparsers.add_parser(name, help=help_text)
        subparser.add_argument("source", type=Path, help=".tex file, directory, or .tar.gz")
        if name in {"tree", "budget"}:
            add_selection_args(subparser)
        if name == "budget":
            add_budget_args(subparser)
        if name == "digest":
            add_digest_args(subparser)
    resume_parser = subparsers.add_parser(
        "resume",
        help="Resume a previous digestion run from its Markdown output path.",
    )
    resume_parser.add_argument("output", type=Path, help="Existing Markdown output path.")
    return parser


def add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--exclude-node",
        type=parse_id_list,
        default=[],
        help="Comma-separated order IDs to exclude, such as 0008,0042.",
    )
    parser.add_argument(
        "--include-node",
        type=parse_id_list,
        default=[],
        help="Comma-separated order IDs to include after type exclusions.",
    )
    parser.add_argument(
        "--split-after",
        type=parse_id_list,
        default=[],
        help="Comma-separated order IDs that force a chunk split after that node.",
    )
    parser.add_argument(
        "--exclude-type",
        type=parse_name_list,
        default=[],
        help="Comma-separated node types to exclude, such as bibliography,figure.",
    )


def add_budget_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-length", type=int, default=10240)
    parser.add_argument("--usable-input", type=int, default=8500)
    parser.add_argument("--reserved-output", type=int, default=768)
    parser.add_argument("--rolling-memory", type=int, default=900)
    parser.add_argument("--instructions", type=int, default=350)


def add_digest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("digestion.md"),
        help="Markdown file that receives streamed chunk outputs and final abstract.",
    )


def parse_id_list(value: str) -> list[int]:
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_name_list(value: str) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "digest":
        return run_digest(args)
    if args.command == "resume":
        return run_resume(args)

    token_counter = load_token_counter(args.tokenizer, args.tokenizer_path)
    root = parse_latex_project(args.source, token_counter)

    if args.command == "ingest":
        nodes = list(root.walk())
        leaves = list(root.leaf_nodes())
        print(f"title: {root.title}")
        print(f"tokenizer: {token_counter.name}")
        print(f"nodes: {len(nodes)}")
        print(f"leaves: {len(leaves)}")
        print(f"tokens: {root.token_count}")
        return 0

    if args.command == "tree":
        apply_selection_edits(
            root,
            exclude_nodes=args.exclude_node,
            include_nodes=args.include_node,
            split_after=args.split_after,
            exclude_types=args.exclude_type,
        )
        print(print_tree(root, include_root=True))
        return 0

    if args.command == "budget":
        apply_selection_edits(
            root,
            exclude_nodes=args.exclude_node,
            include_nodes=args.include_node,
            split_after=args.split_after,
            exclude_types=args.exclude_type,
        )
        budget = TokenBudget(
            cache_length=args.cache_length,
            usable_input_tokens=args.usable_input,
            reserved_output_tokens=args.reserved_output,
            rolling_memory_tokens=args.rolling_memory,
            instruction_tokens=args.instructions,
        )
        chunks = pack_chunks(root, budget)
        print(format_budget_report(chunks, budget))
        return 0

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
        try:
            chunks = terminal_plan(root, budget)
        except curses_error():
            print("Full-screen planner unavailable; falling back to text planner.")
            chunks = interactive_plan(root, budget)
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
        try:
            chunks = terminal_plan(root, budget)
        except curses_error():
            print("Full-screen planner unavailable; falling back to text planner.")
            chunks = interactive_plan(root, budget)
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


def curses_error():
    import curses

    return curses.error
