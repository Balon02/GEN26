from __future__ import annotations

import re
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pylatexenc.latex2text import LatexNodes2Text
from pylatexenc.latexwalker import LatexEnvironmentNode, LatexMacroNode, LatexWalker

from gen26.paper_tree import (
    IncludeStatus,
    PaperNode,
    recompute_parent_totals,
)


SECTION_LEVELS = {
    "part": 0,
    "chapter": 0,
    "section": 1,
    "subsection": 2,
    "subsubsection": 3,
    "paragraph": 4,
}
BLOCK_ENVS = {
    "equation",
    "equation*",
    "align",
    "align*",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "figure",
    "figure*",
    "table",
    "table*",
    "theorem",
    "proof",
    "definition",
    "lemma",
    "proposition",
    "corollary",
    "thebibliography",
}
INPUT_RE = re.compile(r"\\(?:input|include)\{(?P<path>[^}]+)\}")
CAPTION_RE = re.compile(
    r"\\caption(?:\[[^\]]*\])?\{(?P<caption>(?:[^{}]|\{[^{}]*\})*)\}",
    re.DOTALL,
)
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
REF_RE = re.compile(r"\\(?:ref|eqref|autoref|pageref|cite|citep|citet)\{([^}]+)\}")
GRAPHICS_RE = re.compile(
    r"\\includegraphics(?:\[[^\]]*\])?\{(?P<path>[^}]+)\}",
    re.DOTALL,
)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".pdf")


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int:
        ...


@dataclass
class LoadedSource:
    root_dir: Path
    main_file: Path
    text: str
    temp_dir: tempfile.TemporaryDirectory[str] | None = None

    def cleanup(self) -> None:
        if self.temp_dir is not None:
            self.temp_dir.cleanup()


class OrderCounter:
    def __init__(self) -> None:
        self.value = 0

    def next(self) -> int:
        current = self.value
        self.value += 1
        return current


def load_latex_source(path: Path) -> LoadedSource:
    path = path.expanduser().resolve()
    if path.is_file() and path.suffixes[-2:] == [".tar", ".gz"]:
        temp_dir = tempfile.TemporaryDirectory(prefix="gen26-latex-")
        root = Path(temp_dir.name)
        with tarfile.open(path, "r:gz") as archive:
            archive.extractall(root, filter="data")
        main_file = find_main_tex(root)
        text = read_with_inputs(main_file, root, set())
        return LoadedSource(root, main_file, text, temp_dir)

    if path.is_dir():
        main_file = find_main_tex(path)
        return LoadedSource(path, main_file, read_with_inputs(main_file, path, set()))

    if path.is_file() and path.suffix == ".tex":
        root = path.parent
        return LoadedSource(root, path, read_with_inputs(path, root, set()))

    raise ValueError(f"Expected a .tex file, directory, or .tar.gz archive: {path}")


def find_main_tex(root: Path) -> Path:
    candidates = sorted(root.rglob("*.tex"))
    if not candidates:
        raise FileNotFoundError(f"No .tex files found under {root}")

    with_document = []
    for candidate in candidates:
        text = candidate.read_text(encoding="utf-8", errors="replace")
        if "\\begin{document}" in text:
            with_document.append(candidate)
    if with_document:
        return sorted(with_document, key=lambda p: (p.name != "ms.tex", len(p.parts), p.name))[0]
    return candidates[0]


def read_with_inputs(path: Path, root: Path, seen: set[Path]) -> str:
    path = path.resolve()
    if path in seen:
        return ""
    seen.add(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    text = strip_comments(text)

    def replace_input(match: re.Match[str]) -> str:
        rel = match.group("path").strip()
        input_path = (path.parent / rel).with_suffix(".tex")
        if not input_path.exists():
            input_path = (root / rel).with_suffix(".tex")
        if not input_path.exists():
            return f"\n[missing input: {rel}]\n"
        return "\n" + read_with_inputs(input_path, root, seen) + "\n"

    return INPUT_RE.sub(replace_input, text)


def strip_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        cut = None
        for index, char in enumerate(line):
            if char == "%" and (index == 0 or line[index - 1] != "\\"):
                cut = index
                break
        lines.append(line[:cut] if cut is not None else line)
    return "\n".join(lines)


def parse_loaded_source(source: LoadedSource, token_counter: TokenCounter) -> PaperNode:
    order = OrderCounter()
    body = document_body(source.text)
    title = latex_to_text(first_group(source.text, "title") or source.main_file.stem)
    root = PaperNode(
        order=order.next(),
        node_type="paper",
        title=title,
        source_path=source.main_file,
    )

    metadata_parts = []
    for command in ("title", "author", "date"):
        value = first_group(source.text, command)
        if value:
            metadata_parts.append(f"{command}: {latex_to_text(value)}")
    if metadata_parts:
        add_leaf(
            root,
            order,
            "metadata",
            "metadata",
            "\n".join(metadata_parts),
            source.main_file,
            0,
            0,
            token_counter,
        )

    abstract_match = re.search(
        r"\\begin\{abstract\}(?P<body>.*?)\\end\{abstract\}", body, re.DOTALL
    )
    if abstract_match:
        add_leaf(
            root,
            order,
            "abstract",
            "abstract",
            latex_to_text(abstract_match.group("body")),
            source.main_file,
            abstract_match.start(),
            abstract_match.end(),
            token_counter,
        )

    content = re.sub(
        r"\\begin\{abstract\}.*?\\end\{abstract\}",
        "",
        body,
        flags=re.DOTALL,
    )
    parse_blocks(content, root, order, source, token_counter)
    exclude_unsectioned_front_matter(root)
    recompute_parent_totals(root)
    return root


def document_body(text: str) -> str:
    begin = text.find("\\begin{document}")
    end = text.find("\\end{document}")
    if begin != -1:
        begin += len("\\begin{document}")
        return text[begin:end if end != -1 else None]
    return text


def exclude_unsectioned_front_matter(root: PaperNode) -> None:
    has_sections = any(
        child.node_type in {"part", "chapter", "section"} for child in root.children
    )
    if not has_sections:
        return
    for child in root.children:
        if child.node_type == "paragraph":
            child.include_status = IncludeStatus.EXCLUDE


def first_group(text: str, command: str) -> str | None:
    match = re.search(
        rf"\\{command}(?:\[[^\]]*\])?\{{(?P<value>(?:[^{{}}]|\{{[^{{}}]*\}})*)\}}",
        text,
        re.DOTALL,
    )
    return match.group("value") if match else None


def parse_blocks(
    text: str,
    root: PaperNode,
    order: OrderCounter,
    source: LoadedSource,
    token_counter: TokenCounter,
) -> None:
    section_stack: list[tuple[int, PaperNode]] = [(-1, root)]
    paragraph_parts: list[str] = []
    paragraph_start: int | None = None
    nodes, _, _ = LatexWalker(text).get_latex_nodes()

    def flush_paragraphs() -> None:
        nonlocal paragraph_parts, paragraph_start
        if paragraph_parts:
            add_paragraphs(
                current_parent(section_stack),
                order,
                "".join(paragraph_parts),
                source.main_file,
                paragraph_start or 0,
                token_counter,
            )
        paragraph_parts = []
        paragraph_start = None

    for parsed_node in nodes:
        if is_section_node(parsed_node):
            flush_paragraphs()
            kind = parsed_node.macroname
            title = latex_to_text(section_title_source(parsed_node))
            level = SECTION_LEVELS[kind]
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()
            parent = current_parent(section_stack)
            node = PaperNode(
                order=order.next(),
                node_type=kind,
                title=title,
                source_path=source.main_file,
                source_start=parsed_node.pos,
                source_end=parsed_node.pos + parsed_node.len,
            )
            parent.add_child(node)
            section_stack.append((level, node))
            continue

        if is_block_environment(parsed_node):
            flush_paragraphs()
            add_environment(
                current_parent(section_stack),
                order,
                parsed_node.environmentname,
                parsed_node.latex_verbatim(),
                source.root_dir,
                source.main_file,
                parsed_node.pos,
                parsed_node.pos + parsed_node.len,
                token_counter,
            )
            continue

        if paragraph_start is None:
            paragraph_start = parsed_node.pos
        paragraph_parts.append(parsed_node.latex_verbatim())

    flush_paragraphs()


def is_section_node(node) -> bool:
    return isinstance(node, LatexMacroNode) and node.macroname in SECTION_LEVELS


def is_block_environment(node) -> bool:
    if not isinstance(node, LatexEnvironmentNode):
        return False
    return node.environmentname in BLOCK_ENVS


def section_title_source(node: LatexMacroNode) -> str:
    if not node.nodeargd:
        return node.macroname
    for arg in reversed(node.nodeargd.argnlist):
        if arg is not None:
            return arg.latex_verbatim()
    return node.macroname


def current_parent(stack: list[tuple[int, PaperNode]]) -> PaperNode:
    return stack[-1][1]


def add_paragraphs(
    parent: PaperNode,
    order: OrderCounter,
    text: str,
    source_path: Path,
    offset: int,
    token_counter: TokenCounter,
) -> None:
    for paragraph in re.split(r"\n\s*\n", text):
        normalized = latex_to_text(paragraph)
        if len(normalized) < 3:
            continue
        add_leaf(
            parent,
            order,
            "paragraph",
            "paragraph",
            normalized,
            source_path,
            offset,
            offset + len(paragraph),
            token_counter,
        )
        offset += len(paragraph)


def add_environment(
    parent: PaperNode,
    order: OrderCounter,
    env: str,
    raw_block: str,
    root_dir: Path,
    source_path: Path,
    start: int,
    end: int,
    token_counter: TokenCounter,
) -> None:
    caption_match = CAPTION_RE.search(raw_block)
    caption = latex_to_text(caption_match.group("caption")) if caption_match else None
    labels = LABEL_RE.findall(raw_block)
    refs = REF_RE.findall(raw_block)
    image_paths = find_graphics_paths(raw_block, root_dir, source_path)
    base_env = env.rstrip("*")
    if base_env == "thebibliography":
        base_env = "bibliography"
    title = caption or (labels[0] if labels else base_env)
    text = latex_to_text(raw_block)
    node = PaperNode(
        order=order.next(),
        node_type=base_env,
        title=title,
        text=text,
        source_path=source_path,
        source_start=start,
        source_end=end,
        labels=labels,
        references=refs,
        caption=caption,
        image_paths=image_paths,
    )
    if base_env == "bibliography":
        node.include_status = IncludeStatus.EXCLUDE
    count_and_attach(parent, node, token_counter)


def find_graphics_paths(raw_block: str, root_dir: Path, source_path: Path) -> list[Path]:
    paths: list[Path] = []
    for match in GRAPHICS_RE.finditer(raw_block):
        raw_path = match.group("path").strip()
        resolved = resolve_image_path(raw_path, root_dir, source_path)
        if resolved is not None:
            paths.append(resolved)
    return paths


def resolve_image_path(raw_path: str, root_dir: Path, source_path: Path) -> Path | None:
    candidates = [
        (source_path.parent / raw_path).resolve(),
        (root_dir / raw_path).resolve(),
    ]
    for candidate in candidates:
        resolved = resolve_candidate_image_path(candidate)
        if resolved is not None:
            return resolved

    raw_name = Path(raw_path).name
    for found in root_dir.rglob(f"{raw_name}*"):
        if found.suffix.lower() in IMAGE_EXTENSIONS:
            return found.resolve()
    return None


def resolve_candidate_image_path(candidate: Path) -> Path | None:
    if candidate.exists() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
        return candidate
    if not candidate.suffix:
        for extension in IMAGE_EXTENSIONS:
            with_extension = candidate.with_suffix(extension)
            if with_extension.exists():
                return with_extension
    return None


def add_leaf(
    parent: PaperNode,
    order: OrderCounter,
    node_type: str,
    title: str,
    text: str,
    source_path: Path,
    start: int,
    end: int,
    token_counter: TokenCounter,
) -> PaperNode:
    node = PaperNode(
        order=order.next(),
        node_type=node_type,
        title=title,
        text=text,
        source_path=source_path,
        source_start=start,
        source_end=end,
        labels=LABEL_RE.findall(text),
        references=REF_RE.findall(text),
    )
    count_and_attach(parent, node, token_counter)
    return node


def count_and_attach(parent: PaperNode, node: PaperNode, token_counter: TokenCounter) -> None:
    node.token_count = token_counter.count(node.selectable_text())
    parent.add_child(node)


def latex_to_text(text: str) -> str:
    converted = LatexNodes2Text().latex_to_text(text)
    converted = normalize_refs(converted)
    converted = re.sub(r"\n{3,}", "\n\n", converted)
    converted = re.sub(r"[ \t]+", " ", converted)
    converted = re.sub(r"\s+\n", "\n", converted)
    return converted.strip()


def normalize_refs(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        command = match.group(0).split("{", 1)[0].lstrip("\\")
        return f"[{command}: {match.group(1)}]"

    return REF_RE.sub(repl, text)
