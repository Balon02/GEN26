from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gen26.chunking import ChunkPlan
from gen26.images import load_chunk_images
from gen26.paper_tree import IncludeStatus, PaperNode

IMAGE_TOKENS = 256
MEMORY_DELTA_TOKEN_LIMIT = 260
MEMORY_COMPACTION_THRESHOLD = 0.85
MAX_RAW_IMAGES_PER_CHUNK_CALL = 1


@dataclass
class RollingMemory:
    text: str = (
        "Running abstract: empty\n"
        "Key claims: empty\n"
        "Methods and datasets: empty\n"
        "Metrics and results: empty\n"
        "Definitions and notation: empty\n"
        "Limitations and caveats: empty\n"
        "Unresolved ambiguities: empty"
    )


@dataclass
class DigestionResult:
    chunk_summaries: list[str]
    final_abstract: str
    rolling_memory: RollingMemory


def digest_chunks(
    runtime,
    chunks: list[ChunkPlan],
    output_file: Path,
    rolling_memory_token_limit: int = 900,
    run_store=None,
    initial_memory: RollingMemory | None = None,
    initial_summaries: list[str] | None = None,
    append_output: bool = False,
    total_chunks: int | None = None,
) -> DigestionResult:
    memory = initial_memory or RollingMemory()
    chunk_summaries: list[str] = list(initial_summaries or [])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if not append_output:
        output_file.write_text("# Paper Digestion\n\n", encoding="utf-8")
    else:
        append_markdown(output_file, "\n\n# Resume\n\n")
    total = total_chunks or len(chunks)

    for chunk in chunks:
        print(
            f"\n\n=== CHUNK {chunk.index}/{total} "
            f"tokens={chunk.token_count} nodes={len(chunk.nodes)} ===",
            flush=True,
        )
        images, skipped_images = load_chunk_images(
            chunk,
            image_size=runtime.image_size,
        )
        append_markdown(
            output_file,
            f"## Chunk {chunk.index}\n\n"
            f"- Tokens: {chunk.token_count}\n"
            f"- Nodes: {len(chunk.nodes)}\n\n",
        )
        image_summaries: list[str] = []
        chunk_images = images
        if len(images) > MAX_RAW_IMAGES_PER_CHUNK_CALL:
            try:
                image_summaries = digest_images_sequentially(
                    runtime,
                    chunk,
                    images,
                    output_file,
                    run_store,
                )
                chunk_images = []
            except Exception as exc:
                if run_store is not None:
                    run_store.chunk_failed(chunk, exc)
                raise

        prompt = build_chunk_prompt(
            chunk,
            memory,
            chunk_images,
            skipped_images,
            image_summaries,
        )
        prompt_stats = prompt_component_stats(
            runtime,
            prompt,
            chunk,
            memory,
            chunk_images,
            skipped_images,
            image_summaries,
        )
        prompt_stats["source_image_count"] = len(images)
        prompt_stats["image_summary_count"] = len(image_summaries)
        if run_store is not None:
            run_store.chunk_started(chunk, prompt_stats)
        append_markdown(
            output_file,
            "### Prompt Budget\n\n"
            f"- Prompt text tokens: {prompt_stats['prompt_text_tokens']}\n"
            f"- Chunk text tokens: {prompt_stats['chunk_text_tokens']}\n"
            f"- Rolling memory tokens: {prompt_stats['rolling_memory_tokens']}\n"
            f"- Source images: {prompt_stats['source_image_count']}\n"
            f"- Raw images in chunk call: {prompt_stats['image_count']}\n"
            f"- Image summaries: {prompt_stats['image_summary_count']}\n"
            f"- Image tokens: {prompt_stats['image_tokens']}\n"
            f"- Estimated total input: {prompt_stats['estimated_total_input_tokens']}\n\n"
            "### Model Output\n\n",
        )
        try:
            check_prompt_budget(runtime, prompt_stats, chunk)
            with output_file.open("a", encoding="utf-8") as stream_file:
                response = runtime.chat(
                    prompt,
                    images=[image.array for image in chunk_images],
                    stream_file=stream_file,
                )
        except Exception as exc:
            if run_store is not None:
                run_store.chunk_failed(chunk, exc)
            raise
        if not response:
            response = "[streamed response was not returned by ChatSampler]"
            append_markdown(output_file, response)
        try:
            local_summary = extract_section(response, "LOCAL_SUMMARY") or response
            chunk_summaries.append(bound_text(runtime, local_summary, 350))

            memory_delta = extract_section(response, "MEMORY_DELTA")
            if useful_memory_delta(memory_delta):
                memory.text = append_memory_delta(
                    memory.text,
                    chunk.index,
                    bound_text(runtime, memory_delta, MEMORY_DELTA_TOKEN_LIMIT),
                )

            if should_compact_memory(runtime, memory.text, rolling_memory_token_limit):
                memory.text = compact_memory(
                    runtime,
                    memory.text,
                    rolling_memory_token_limit,
                    output_file,
                    run_store,
                )
        except Exception as exc:
            if run_store is not None:
                run_store.chunk_failed(chunk, exc)
            raise

        append_markdown(
            output_file,
            "\n\n### Rolling Memory After Chunk\n\n"
            f"{memory.text}\n\n",
        )
        if run_store is not None:
            run_store.chunk_completed(chunk, chunk_summaries[-1], memory.text)

    print("\n\n=== FINAL ABSTRACT ===", flush=True)
    final_prompt = build_final_prompt_that_fits(runtime, memory, chunk_summaries)
    final_tokens = runtime.count_tokens(final_prompt)
    if run_store is not None:
        run_store.final_started(final_tokens)
    append_markdown(output_file, "## Final Pass\n\n")
    try:
        check_text_prompt_budget(runtime, final_prompt, "final abstract")
        with output_file.open("a", encoding="utf-8") as stream_file:
            final_abstract = runtime.chat(final_prompt, stream_file=stream_file)
    except Exception as exc:
        if run_store is not None:
            run_store.final_failed(exc)
        raise
    if not final_abstract:
        final_abstract = "[streamed final abstract was not returned by ChatSampler]"
        append_markdown(output_file, final_abstract)

    append_markdown(
        output_file,
        "\n\n## Final Rolling Memory\n\n"
        f"{memory.text}\n",
    )
    if run_store is not None:
        run_store.finish(final_abstract)

    return DigestionResult(
        chunk_summaries=chunk_summaries,
        final_abstract=final_abstract,
        rolling_memory=memory,
    )


def build_chunk_prompt(
    chunk: ChunkPlan,
    memory: RollingMemory,
    images,
    skipped_images: list[str],
    image_summaries: list[str] | None = None,
) -> str:
    image_lines = []
    for index, image in enumerate(images, start=1):
        image_lines.append(f"Image {index}: {image.path.name} <|image|>")
    for skipped in skipped_images:
        image_lines.append(f"Skipped image: {skipped}")
    if image_summaries:
        image_lines.append("Sequential image readings:")
        image_lines.extend(image_summaries)

    return (
        "You are digesting a research paper in ordered chunks. Stay faithful to "
        "the supplied source. Do not invent claims, metrics, datasets, or "
        "limitations.\n\n"
        "Current rolling memory:\n"
        f"{memory.text}\n\n"
        f"Current chunk {chunk.index}:\n"
        f"{format_chunk_text(chunk)}\n\n"
        "Relevant visual payloads:\n"
        f"{chr(10).join(image_lines) if image_lines else 'none'}\n\n"
        "The parser may label ordinary prose blocks as paragraph nodes. Treat "
        "their text as normal paper content, but do not discuss parser labels "
        "or LaTeX structure unless it matters scientifically.\n\n"
        "Return exactly these sections:\n"
        "LOCAL_SUMMARY:\n"
        "MEMORY_DELTA:\n"
        "IMPORTANT_CLAIMS_RESULTS:\n"
        "LIMITATIONS_OR_UNCERTAINTIES:\n"
        "FIGURES_TABLES_USED:\n"
        "\n"
        "MEMORY_DELTA must contain only new durable facts that should survive "
        "to later chunks. Do not rewrite the current rolling memory. Do not "
        "copy source text. Write 'none' if this chunk adds no durable facts.\n"
    )


def build_final_prompt(
    memory: RollingMemory,
    chunk_summaries: list[str],
) -> str:
    summaries = "\n\n".join(
        f"CHUNK {index} SUMMARY:\n{summary}"
        for index, summary in enumerate(chunk_summaries, start=1)
    )


def digest_images_sequentially(
    runtime,
    chunk: ChunkPlan,
    images,
    output_file: Path,
    run_store=None,
) -> list[str]:
    append_markdown(
        output_file,
        "### Image Prepass\n\n"
        f"{len(images)} images found. Reading them one at a time to avoid "
        "multi-image vision memory pressure.\n\n",
    )
    summaries: list[str] = []
    context = bound_text(runtime, format_chunk_text(chunk), 900)
    for index, image in enumerate(images, start=1):
        prompt = build_image_prompt(chunk, image.path.name, context)
        prompt_tokens = runtime.count_tokens(prompt)
        if run_store is not None:
            run_store.image_started(chunk.index, index, image.path.name, prompt_tokens)
        append_markdown(
            output_file,
            f"#### Image {index}: {image.path.name}\n\n"
            f"- Prompt text tokens: {prompt_tokens}\n"
            f"- Image tokens: {IMAGE_TOKENS}\n\n",
        )
        try:
            check_text_prompt_budget(runtime, prompt, f"image {index} prepass")
            with output_file.open("a", encoding="utf-8") as stream_file:
                response = runtime.chat(
                    prompt,
                    images=[image.array],
                    stream_file=stream_file,
                )
        except Exception as exc:
            if run_store is not None:
                run_store.image_failed(chunk.index, index, image.path.name, exc)
            raise
        summary = extract_section(response or "", "IMAGE_SUMMARY") or response or ""
        summary = bound_text(runtime, summary.strip(), 180)
        summaries.append(f"Image {index} ({image.path.name}): {summary}")
        if run_store is not None:
            run_store.image_completed(chunk.index, index, image.path.name, len(summary))
        append_markdown(output_file, "\n\n")
    return summaries


def build_image_prompt(chunk: ChunkPlan, image_name: str, context: str) -> str:
    return (
        "Read this research-paper figure/table image. Use the surrounding text "
        "only as context; describe what is actually visible. Keep the summary "
        "short and factual, and note if the image is unreadable.\n\n"
        f"Chunk {chunk.index}: {chunk.title()}\n"
        f"Image file: {image_name}\n\n"
        "Surrounding chunk text excerpt:\n"
        f"{context}\n\n"
        "Image payload:\n"
        "<|image|>\n\n"
        "Return exactly this section:\n"
        "IMAGE_SUMMARY:\n"
    )
    return (
        "Write the final abstract from the accumulated digestion state. Stay "
        "faithful to the paper and avoid unsupported claims.\n\n"
        "Final rolling memory:\n"
        f"{memory.text}\n\n"
        "Ordered local summaries:\n"
        f"{summaries}\n\n"
        "Return exactly these sections:\n"
        "FINAL_ABSTRACT:\n"
        "SHORT_SUMMARY:\n"
        "STRUCTURED_NOTES:\n"
    )


def build_final_prompt_that_fits(
    runtime,
    memory: RollingMemory,
    chunk_summaries: list[str],
) -> str:
    for token_limit in (350, 220, 140, 90):
        bounded = [bound_text(runtime, summary, token_limit) for summary in chunk_summaries]
        prompt = build_final_prompt(memory, bounded)
        if runtime.count_tokens(prompt) <= runtime.safe_input_tokens:
            return prompt
    raise ValueError(
        "Final abstract prompt does not fit even after compacting chunk summaries."
    )


def format_chunk_text(chunk: ChunkPlan) -> str:
    parts = []
    for node in chunk.nodes:
        parts.append(format_node_subtree(node))
    return "\n\n".join(parts)


def format_node_subtree(node: PaperNode) -> str:
    lines = [format_node_heading(node)]

    if not node.children:
        text = model_text_for_node(node)
        if text:
            lines.append(text)
        return "\n".join(lines)

    leaves = list(selected_leaves(node))
    if not leaves:
        text = node.selectable_text()
        if text:
            lines.append(text)
        return "\n".join(lines)

    for leaf in leaves:
        lines.append("")
        heading = format_node_heading(leaf)
        if heading:
            lines.append(heading)
        leaf_text = model_text_for_node(leaf)
        if leaf_text:
            lines.append(leaf_text)
    return "\n".join(lines)


def format_node_heading(node: PaperNode) -> str:
    if node.node_type == "paragraph" and node.title == "paragraph":
        return ""
    return f"[{node.order:04d}] {node.node_type.upper()} - {node.display_label()}"


def model_text_for_node(node: PaperNode) -> str:
    parts = []
    if node.caption:
        parts.append(f"Caption: {node.caption}")
    if node.image_paths:
        images = ", ".join(path.name for path in node.image_paths)
        parts.append(f"Images: {images}")
    if node.text:
        parts.append(node.text)
    return "\n\n".join(parts).strip()


def selected_leaves(node: PaperNode):
    if node.include_status == IncludeStatus.EXCLUDE:
        return
    if not node.children:
        yield node
        return
    for child in node.children:
        yield from selected_leaves(child)


def extract_section(text: str, section_name: str) -> str:
    marker = f"{section_name}:"
    start = text.find(marker)
    if start == -1:
        return ""
    start += len(marker)
    next_start = len(text)
    for candidate in (
        "LOCAL_SUMMARY:",
        "MEMORY_DELTA:",
        "UPDATED_ROLLING_MEMORY:",
        "COMPACTED_ROLLING_MEMORY:",
        "IMPORTANT_CLAIMS_RESULTS:",
        "LIMITATIONS_OR_UNCERTAINTIES:",
        "FIGURES_TABLES_USED:",
    ):
        index = text.find(candidate, start)
        if index != -1:
            next_start = min(next_start, index)
    return text[start:next_start].strip()


def bound_text(runtime, text: str, token_limit: int) -> str:
    if runtime.count_tokens(text) <= token_limit:
        return text
    lines = text.splitlines()
    while lines and runtime.count_tokens("\n".join(lines)) > token_limit:
        lines.pop()
    return "\n".join(lines).strip()


def useful_memory_delta(delta: str) -> bool:
    normalized = delta.strip().lower()
    return bool(normalized and normalized not in {"none", "n/a", "no new durable facts"})


def append_memory_delta(memory_text: str, chunk_index: int, delta: str) -> str:
    if not delta.strip():
        return memory_text
    return (
        memory_text.rstrip()
        + f"\n\nChunk {chunk_index} durable additions:\n"
        + delta.strip()
    )


def should_compact_memory(runtime, memory_text: str, token_limit: int) -> bool:
    return runtime.count_tokens(memory_text) >= round(
        token_limit * MEMORY_COMPACTION_THRESHOLD
    )


def compact_memory(
    runtime,
    memory_text: str,
    token_limit: int,
    output_file: Path,
    run_store=None,
) -> str:
    before_tokens = runtime.count_tokens(memory_text)
    target_tokens = max(200, round(token_limit * 0.65))
    print(
        "\n=== COMPACTING ROLLING MEMORY "
        f"tokens={before_tokens} target~{target_tokens} ===",
        flush=True,
    )
    append_markdown(
        output_file,
        "\n\n### Rolling Memory Compaction\n\n"
        f"- Before tokens: {before_tokens}\n"
        f"- Target tokens: {target_tokens}\n\n"
        "### Compaction Output\n\n",
    )
    if run_store is not None:
        run_store.memory_compaction_started(before_tokens, target_tokens)

    prompt = build_memory_compaction_prompt(memory_text, target_tokens)
    check_text_prompt_budget(runtime, prompt, "rolling memory compaction")
    with output_file.open("a", encoding="utf-8") as stream_file:
        response = runtime.chat(prompt, stream_file=stream_file)
    compacted = extract_section(response or "", "COMPACTED_ROLLING_MEMORY") or response or ""
    compacted = bound_text(runtime, compacted.strip(), target_tokens)
    after_tokens = runtime.count_tokens(compacted)
    if run_store is not None:
        run_store.memory_compaction_completed(before_tokens, after_tokens)
    append_markdown(
        output_file,
        "\n\n### Compacted Rolling Memory\n\n"
        f"{compacted}\n\n",
    )
    return compacted or bound_text(runtime, memory_text, target_tokens)


def build_memory_compaction_prompt(memory_text: str, target_tokens: int) -> str:
    return (
        "Compact the rolling memory for a sequential research-paper digestion. "
        "Preserve durable facts, definitions, methods, datasets, metrics, "
        "results, visual evidence, and limitations. Remove duplicated wording, "
        "parser artifacts, source-text copies, and stale local details. "
        f"Keep the result under about {target_tokens} tokens.\n\n"
        "Rolling memory to compact:\n"
        f"{memory_text}\n\n"
        "Return exactly this section:\n"
        "COMPACTED_ROLLING_MEMORY:\n"
    )


def prompt_component_stats(
    runtime,
    prompt: str,
    chunk: ChunkPlan,
    memory: RollingMemory,
    images,
    skipped_images: list[str],
) -> dict[str, int]:
    chunk_text = format_chunk_text(chunk)
    return {
        "prompt_text_tokens": runtime.count_tokens(prompt),
        "chunk_text_tokens": runtime.count_tokens(chunk_text),
        "rolling_memory_tokens": runtime.count_tokens(memory.text),
        "image_count": len(images),
        "skipped_image_count": len(skipped_images),
        "image_tokens": len(images) * IMAGE_TOKENS,
        "estimated_total_input_tokens": runtime.count_tokens(prompt)
        + len(images) * IMAGE_TOKENS,
        "safe_input_tokens": runtime.safe_input_tokens,
    }


def check_prompt_budget(runtime, prompt_stats: dict[str, int], chunk: ChunkPlan) -> None:
    if prompt_stats["estimated_total_input_tokens"] > runtime.safe_input_tokens:
        raise ValueError(
            f"Chunk {chunk.index} exceeds safe prompt budget: "
            f"text={prompt_stats['prompt_text_tokens']}, "
            f"images={prompt_stats['image_tokens']}, "
            f"total={prompt_stats['estimated_total_input_tokens']}, "
            f"limit={runtime.safe_input_tokens}. "
            "Split this node further in the planner."
        )


def check_text_prompt_budget(runtime, prompt: str, label: str) -> None:
    tokens = runtime.count_tokens(prompt)
    if tokens > runtime.safe_input_tokens:
        raise ValueError(
            f"The {label} prompt has {tokens} tokens, over the safe limit "
            f"{runtime.safe_input_tokens}."
        )


def append_markdown(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(text)
