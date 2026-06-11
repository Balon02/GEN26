from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable


class IncludeStatus(str, Enum):
    INCLUDE = "include"
    EXCLUDE = "exclude"


class DigestMode(str, Enum):
    AUTO = "auto"
    WHOLE = "whole"
    SPLIT = "split"


@dataclass
class PaperNode:
    order: int
    node_type: str
    title: str
    text: str = ""
    source_path: Path | None = None
    source_start: int | None = None
    source_end: int | None = None
    labels: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    caption: str | None = None
    image_paths: list[Path] = field(default_factory=list)
    token_count: int = 0
    estimated_output_tokens: int = 0
    include_status: IncludeStatus = IncludeStatus.INCLUDE
    digest_mode: DigestMode = DigestMode.AUTO
    force_split_after: bool = False
    children: list["PaperNode"] = field(default_factory=list)

    def add_child(self, node: "PaperNode") -> "PaperNode":
        self.children.append(node)
        return node

    def walk(self) -> Iterable["PaperNode"]:
        yield self
        for child in self.children:
            yield from child.walk()

    def selected_walk(self) -> Iterable["PaperNode"]:
        if self.include_status == IncludeStatus.EXCLUDE:
            return
        yield self
        for child in self.children:
            yield from child.selected_walk()

    def leaf_nodes(self) -> Iterable["PaperNode"]:
        if not self.children:
            yield self
            return
        for child in self.children:
            yield from child.leaf_nodes()

    def selectable_text(self) -> str:
        parts = []
        if self.title:
            parts.append(f"{self.node_type.upper()}: {self.title}")
        if self.caption:
            parts.append(f"Caption: {self.caption}")
        if self.image_paths:
            images = ", ".join(path.name for path in self.image_paths)
            parts.append(f"Images: {images}")
        if self.text:
            parts.append(self.text)
        return "\n\n".join(parts).strip()

    def display_label(self, max_chars: int = 96) -> str:
        title = self.title or self.caption or self.text.replace("\n", " ")[:48]
        title = " ".join(title.split())
        if not title:
            return "(untitled)"
        if len(title) <= max_chars:
            return title
        return title[: max_chars - 1].rstrip() + "..."


def recompute_parent_totals(node: PaperNode) -> int:
    if not node.children:
        if node.include_status == IncludeStatus.EXCLUDE:
            return 0
        return node.token_count
    node.token_count = sum(recompute_parent_totals(child) for child in node.children)
    node.estimated_output_tokens = sum(
        child.estimated_output_tokens
        for child in node.children
        if child.include_status != IncludeStatus.EXCLUDE
    )
    if node.include_status == IncludeStatus.EXCLUDE:
        return 0
    return node.token_count


def estimate_output_budget(input_tokens: int) -> int:
    if input_tokens <= 0:
        return 0
    return min(512, max(64, round(input_tokens * 0.18)))


def print_tree(node: PaperNode, include_root: bool = True) -> str:
    lines: list[str] = []

    def visit(current: PaperNode, depth: int) -> None:
        marker = "!" if current.force_split_after else " "
        status = current.include_status.value
        source = ""
        if current.source_path is not None:
            source = f" [{current.source_path.name}]"
        lines.append(
            f"{'  ' * depth}{marker} {current.order:04d} "
            f"{current.node_type:<12} {current.token_count:>5} tok "
            f"out~{current.estimated_output_tokens:<4} {status:<7} "
            f"{current.digest_mode.value:<5} "
            f"{current.display_label()}{source}"
        )
        for child in current.children:
            visit(child, depth + 1)

    if include_root:
        visit(node, 0)
    else:
        for child_node in node.children:
            visit(child_node, 0)
    return "\n".join(lines)
