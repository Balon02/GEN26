from __future__ import annotations

from dataclasses import dataclass, field

from gen26.paper_tree import (
    DigestMode,
    IncludeStatus,
    PaperNode,
)


NODE_LEVELS = {
    "metadata": 0,
    "abstract": 0,
    "section": 1,
    "subsection": 2,
    "subsubsection": 3,
    "paragraph": 4,
    "equation": 4,
    "figure": 4,
    "table": 4,
    "theorem": 4,
    "proof": 4,
    "definition": 4,
    "bibliography": 4,
}
DEFAULT_RESERVED_OUTPUT_TOKENS = 768
DEFAULT_ROLLING_MEMORY_TOKENS = 900
DEFAULT_INSTRUCTION_TOKENS = 350


def scaled_token_count(value: int, context_scale: float) -> int:
    if context_scale <= 0:
        raise ValueError("context_scale must be greater than zero.")
    return max(1, round(value * context_scale))


@dataclass
class TokenBudget:
    cache_length: int = 10240
    usable_input_tokens: int = 8500
    reserved_output_tokens: int = DEFAULT_RESERVED_OUTPUT_TOKENS
    rolling_memory_tokens: int = DEFAULT_ROLLING_MEMORY_TOKENS
    instruction_tokens: int = DEFAULT_INSTRUCTION_TOKENS
    context_scale: float = 1.0

    @property
    def chunk_text_tokens(self) -> int:
        return (
            self.usable_input_tokens
            - self.rolling_memory_tokens
            - self.instruction_tokens
        )


def make_token_budget(
    cache_length: int,
    usable_input_tokens: int,
    context_scale: float = 1.0,
) -> TokenBudget:
    return TokenBudget(
        cache_length=cache_length,
        usable_input_tokens=usable_input_tokens,
        reserved_output_tokens=scaled_token_count(
            DEFAULT_RESERVED_OUTPUT_TOKENS,
            context_scale,
        ),
        rolling_memory_tokens=scaled_token_count(
            DEFAULT_ROLLING_MEMORY_TOKENS,
            context_scale,
        ),
        instruction_tokens=scaled_token_count(
            DEFAULT_INSTRUCTION_TOKENS,
            context_scale,
        ),
        context_scale=context_scale,
    )


@dataclass
class ChunkPlan:
    index: int
    nodes: list[PaperNode] = field(default_factory=list)
    token_count: int = 0

    def title(self) -> str:
        if not self.nodes:
            return "(empty)"
        first = self.nodes[0].display_label()
        last = self.nodes[-1].display_label()
        return first if first == last else f"{first} -> {last}"


def pack_chunks(
    root: PaperNode,
    budget: TokenBudget,
    default_level: str = "subsection",
) -> list[ChunkPlan]:
    units = list(plan_digest_units(root, budget, default_level))
    chunks: list[ChunkPlan] = []
    for index, unit in enumerate(units, start=1):
        chunks.append(
            ChunkPlan(
                index=index,
                nodes=[unit],
                token_count=unit.token_count,
            )
        )
    return chunks


def plan_digest_units(
    root: PaperNode,
    budget: TokenBudget,
    default_level: str = "subsection",
):
    target_level = parse_default_level(default_level)
    limit = budget.chunk_text_tokens
    if limit <= 0:
        raise ValueError(
            "Token budget leaves no room for chunk text after instructions and memory."
        )
    yield from walk_digest_units(root, target_level, limit, excluded=False)


def walk_digest_units(
    node: PaperNode,
    target_level: int,
    token_limit: int,
    excluded: bool,
):
    excluded = excluded or node.include_status == IncludeStatus.EXCLUDE
    if excluded:
        return

    if node.node_type == "paper":
        for child in node.children:
            yield from walk_digest_units(child, target_level, token_limit, excluded)
        return

    if node.digest_mode == DigestMode.WHOLE:
        if node.token_count > token_limit:
            raise ValueError(
                f"Node {node.order:04d} was marked whole but has "
                f"{node.token_count} tokens, over the chunk text limit {token_limit}."
            )
        yield node
        return

    if node.digest_mode == DigestMode.SPLIT:
        if not node.children:
            yield node
            return
        for child in node.children:
            yield from walk_digest_units(child, target_level, token_limit, excluded)
        return

    node_level = NODE_LEVELS.get(node.node_type, 4)
    should_digest_here = (
        node_level == 0 or node_level == target_level or not node.children
    ) and node.token_count <= token_limit
    if should_digest_here or not node.children:
        if node.token_count > token_limit and node.children:
            for child in node.children:
                yield from walk_digest_units(child, target_level, token_limit, excluded)
        else:
            yield node
        return

    for child in node.children:
        yield from walk_digest_units(child, target_level, token_limit, excluded)


def parse_default_level(level: str) -> int:
    if level not in NODE_LEVELS:
        allowed = ", ".join(sorted(NODE_LEVELS))
        raise ValueError(f"Unknown digest level {level!r}. Expected one of: {allowed}")
    return NODE_LEVELS[level]


def format_budget_report(chunks: list[ChunkPlan], budget: TokenBudget) -> str:
    lines = [
        "Token budget",
        f"  cache length:          {budget.cache_length}",
        f"  usable input:          {budget.usable_input_tokens}",
        f"  instructions:          {budget.instruction_tokens}",
        f"  rolling memory:        {budget.rolling_memory_tokens}",
        f"  reserved output:       {budget.reserved_output_tokens}",
        f"  context scale:         {budget.context_scale}",
        f"  chunk text limit:      {budget.chunk_text_tokens}",
        "",
        "Chunks",
    ]
    if not chunks:
        lines.append("  (none)")
        return "\n".join(lines)

    for chunk in chunks:
        overflow = " OVER" if chunk.token_count > budget.chunk_text_tokens else ""
        lines.append(
            f"  {chunk.index:02d}. {chunk.token_count:>5} tok{overflow}  "
            f"{len(chunk.nodes):>3} nodes  {chunk.title()}"
        )
    return "\n".join(lines)
