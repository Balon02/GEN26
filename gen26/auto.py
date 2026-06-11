from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from gen26.chunking import ChunkPlan, TokenBudget, format_budget_report, pack_chunks
from gen26.latex_parser import load_latex_source, parse_loaded_source
from gen26.paper_tree import DigestMode, IncludeStatus, PaperNode

if TYPE_CHECKING:
    from gen26.digestion import DigestionResult


class RuntimeTokenCounter:
    name = "gemma3"

    def __init__(self, runtime) -> None:
        self.runtime = runtime

    def count(self, text: str) -> int:
        return self.runtime.count_tokens(text)


def digest_auto(source: str | Path, output: str | Path) -> DigestionResult:
    """Digest a LaTeX paper without an interactive planner.

    The automatic plan bundles each included top-level child of the paper root
    into one chunk. Bibliography and other parser defaults remain unchanged. If
    any top-level chunk exceeds the available chunk text budget, this function
    raises before starting model generation.
    """

    from gen26.digestion import digest_chunks
    from gen26.gemma_runtime import GemmaDigestRuntime
    from gen26.run_store import RunStore

    source_path = Path(source)
    output_path = Path(output)
    runtime = GemmaDigestRuntime()
    loaded_source = load_latex_source(source_path)
    try:
        root = parse_loaded_source(loaded_source, RuntimeTokenCounter(runtime))
        budget = TokenBudget(
            cache_length=runtime.cache_length,
            usable_input_tokens=runtime.safe_input_tokens,
        )
        chunks = plan_top_level_chunks(root, budget)
        print(format_budget_report(chunks, budget), flush=True)

        store = RunStore(output_path)
        store.create(source_path.resolve(), runtime, budget, root, chunks)
        return digest_chunks(
            runtime,
            chunks,
            output_file=output_path,
            rolling_memory_token_limit=budget.rolling_memory_tokens,
            run_store=store,
        )
    finally:
        loaded_source.cleanup()


def plan_top_level_chunks(root: PaperNode, budget: TokenBudget) -> list[ChunkPlan]:
    """Bundle each included top-level paper node and validate chunk sizes."""

    for child in root.children:
        if child.include_status != IncludeStatus.EXCLUDE:
            child.digest_mode = DigestMode.WHOLE

    chunks = pack_chunks(root, budget)
    too_large = [
        chunk
        for chunk in chunks
        if chunk.token_count > budget.chunk_text_tokens
    ]
    if too_large:
        lines = [
            "Automatic top-level plan contains over-budget chunks:",
            f"chunk text limit: {budget.chunk_text_tokens}",
        ]
        for chunk in too_large:
            lines.append(
                f"- chunk {chunk.index:02d}: {chunk.token_count} tokens, "
                f"{chunk.title()}"
            )
        raise ValueError("\n".join(lines))
    return chunks
