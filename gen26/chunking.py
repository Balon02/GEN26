from __future__ import annotations

from dataclasses import dataclass, field

from gen26.paper_tree import (
    DigestMode,
    IncludeStatus,
    PaperNode,
    recompute_parent_totals,
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


@dataclass
class TokenBudget:
    cache_length: int = 10240
    usable_input_tokens: int = 8500
    reserved_output_tokens: int = 768
    rolling_memory_tokens: int = 900
    instruction_tokens: int = 350

    @property
    def chunk_text_tokens(self) -> int:
        return (
            self.usable_input_tokens
            - self.rolling_memory_tokens
            - self.instruction_tokens
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


def pack_chunks_max_fill(root: PaperNode, budget: TokenBudget) -> list[ChunkPlan]:
    limit = budget.chunk_text_tokens
    if limit <= 0:
        raise ValueError(
            "Token budget leaves no room for chunk text after instructions and memory."
        )

    chunks: list[ChunkPlan] = []
    current = ChunkPlan(index=1)
    forced_split_leaf_orders = split_leaf_orders(root)

    for node in selected_leaf_nodes(root):
        if node.token_count > limit:
            if current.nodes:
                chunks.append(current)
                current = ChunkPlan(index=len(chunks) + 1)
            chunks.append(
                ChunkPlan(
                    index=len(chunks) + 1,
                    nodes=[node],
                    token_count=node.token_count,
                )
            )
            continue

        if current.nodes and current.token_count + node.token_count > limit:
            chunks.append(current)
            current = ChunkPlan(index=len(chunks) + 1)

        current.nodes.append(node)
        current.token_count += node.token_count

        if node.order in forced_split_leaf_orders and current.nodes:
            chunks.append(current)
            current = ChunkPlan(index=len(chunks) + 1)

    if current.nodes:
        chunks.append(current)

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


def split_leaf_orders(root: PaperNode) -> set[int]:
    orders: set[int] = set()
    for node in root.walk():
        if not node.force_split_after:
            continue
        leaves = list(node.leaf_nodes())
        if leaves:
            orders.add(leaves[-1].order)
    return orders


def selected_leaf_nodes(root: PaperNode):
    def visit(node: PaperNode, excluded: bool):
        excluded = excluded or node.include_status == IncludeStatus.EXCLUDE
        if not node.children:
            if not excluded:
                yield node
            return
        for child in node.children:
            yield from visit(child, excluded)

    yield from visit(root, False)


def find_node(root: PaperNode, order: int) -> PaperNode:
    for node in root.walk():
        if node.order == order:
            return node
    raise ValueError(f"No node with order id {order:04d}")


def apply_selection_edits(
    root: PaperNode,
    exclude_nodes: list[int] | None = None,
    include_nodes: list[int] | None = None,
    split_after: list[int] | None = None,
    exclude_types: list[str] | None = None,
    whole_nodes: list[int] | None = None,
    split_nodes: list[int] | None = None,
) -> None:
    excluded_types = {node_type.lower() for node_type in exclude_types or []}
    for node in root.walk():
        if node.node_type.lower() in excluded_types:
            node.include_status = IncludeStatus.EXCLUDE

    for order in exclude_nodes or []:
        find_node(root, order).include_status = IncludeStatus.EXCLUDE
    for order in include_nodes or []:
        find_node(root, order).include_status = IncludeStatus.INCLUDE
    for order in split_after or []:
        find_node(root, order).force_split_after = True
    for order in whole_nodes or []:
        find_node(root, order).digest_mode = DigestMode.WHOLE
    for order in split_nodes or []:
        find_node(root, order).digest_mode = DigestMode.SPLIT
    recompute_parent_totals(root)


def format_budget_report(chunks: list[ChunkPlan], budget: TokenBudget) -> str:
    lines = [
        "Token budget",
        f"  cache length:          {budget.cache_length}",
        f"  usable input:          {budget.usable_input_tokens}",
        f"  instructions:          {budget.instruction_tokens}",
        f"  rolling memory:        {budget.rolling_memory_tokens}",
        f"  reserved output:       {budget.reserved_output_tokens}",
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
