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
    include_status: IncludeStatus = IncludeStatus.INCLUDE
    digest_mode: DigestMode = DigestMode.AUTO
    children: list["PaperNode"] = field(default_factory=list)

    def add_child(self, node: "PaperNode") -> "PaperNode":
        self.children.append(node)
        return node

    def walk(self) -> Iterable["PaperNode"]:
        yield self
        for child in self.children:
            yield from child.walk()

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
    if node.include_status == IncludeStatus.EXCLUDE:
        return 0
    return node.token_count
