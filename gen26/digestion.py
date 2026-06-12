from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gen26.chunking import ChunkPlan
from gen26.chunking import scaled_token_count
from gen26.images import load_chunk_images
from gen26.paper_tree import IncludeStatus, PaperNode

IMAGE_TOKENS = 256
MEMORY_DELTA_TOKEN_LIMIT = 260
MAX_RAW_IMAGES_PER_CHUNK_CALL = 1
CHUNK_SUMMARY_TOKEN_LIMIT = 350
IMAGE_CONTEXT_TOKEN_LIMIT = 900
IMAGE_SUMMARY_TOKEN_LIMIT = 180
FINAL_SUMMARY_TOKEN_LIMIT = 700
FINAL_IMAGE_NOTE_TOKEN_LIMIT = 1200
FINAL_MEMORY_DELTA_TOKEN_LIMIT = 3000
EMERGENCY_MEMORY_TARGET_TOKENS = 2200
FINAL_SUMMARY_FALLBACK_LIMITS = (500, 350, 220)
FINAL_COMPACTED_SUMMARY_LIMITS = (350, 220, 140)
FINAL_COMPACTED_MEMORY_DELTA_TOKEN_LIMIT = 1800
FINAL_COMPACTED_IMAGE_NOTE_TOKEN_LIMIT = 800
MEMORY_COMPACTION_MIN_TARGET_TOKENS = 200


@dataclass
class DigestionLimits:
    memory_delta_tokens: int
    chunk_summary_tokens: int
    image_context_tokens: int
    image_summary_tokens: int
    final_summary_tokens: int
    final_image_note_tokens: int
    final_memory_delta_tokens: int
    emergency_memory_target_tokens: int
    final_summary_fallback_tokens: tuple[int, ...]
    final_compacted_summary_tokens: tuple[int, ...]
    final_compacted_memory_delta_tokens: int
    final_compacted_image_note_tokens: int
    memory_compaction_min_target_tokens: int


def make_digestion_limits(context_scale: float = 1.0) -> DigestionLimits:
    def scale(value: int) -> int:
        return scaled_token_count(value, context_scale)

    return DigestionLimits(
        memory_delta_tokens=scale(MEMORY_DELTA_TOKEN_LIMIT),
        chunk_summary_tokens=scale(CHUNK_SUMMARY_TOKEN_LIMIT),
        image_context_tokens=scale(IMAGE_CONTEXT_TOKEN_LIMIT),
        image_summary_tokens=scale(IMAGE_SUMMARY_TOKEN_LIMIT),
        final_summary_tokens=scale(FINAL_SUMMARY_TOKEN_LIMIT),
        final_image_note_tokens=scale(FINAL_IMAGE_NOTE_TOKEN_LIMIT),
        final_memory_delta_tokens=scale(FINAL_MEMORY_DELTA_TOKEN_LIMIT),
        emergency_memory_target_tokens=scale(EMERGENCY_MEMORY_TARGET_TOKENS),
        final_summary_fallback_tokens=tuple(
            scale(value) for value in FINAL_SUMMARY_FALLBACK_LIMITS
        ),
        final_compacted_summary_tokens=tuple(
            scale(value) for value in FINAL_COMPACTED_SUMMARY_LIMITS
        ),
        final_compacted_memory_delta_tokens=scale(
            FINAL_COMPACTED_MEMORY_DELTA_TOKEN_LIMIT
        ),
        final_compacted_image_note_tokens=scale(
            FINAL_COMPACTED_IMAGE_NOTE_TOKEN_LIMIT
        ),
        memory_compaction_min_target_tokens=scale(
            MEMORY_COMPACTION_MIN_TARGET_TOKENS
        ),
    )


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
    image_notes: list[str]
    memory_deltas: list[str]


def digest_chunks(
    runtime,
    chunks: list[ChunkPlan],
    output_file: Path,
    rolling_memory_token_limit: int = 900,
    context_scale: float = 1.0,
    run_store=None,
    initial_memory: RollingMemory | None = None,
    initial_summaries: list[str] | None = None,
    append_output: bool = False,
    total_chunks: int | None = None,
) -> DigestionResult:
    limits = make_digestion_limits(context_scale)
    memory = initial_memory or RollingMemory()
    chunk_summaries: list[str] = list(initial_summaries or [])
    memory_deltas: list[str] = load_initial_memory_deltas(run_store, chunk_summaries)
    image_notes: list[str] = load_initial_image_notes(run_store)
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
                    limits,
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
                    images=image_arrays_or_none(chunk_images),
                    stream_file=stream_file,
                )
        except Exception as exc:
            if run_store is not None:
                run_store.chunk_failed(chunk, exc)
            raise
        if not response:
            response = "[streamed response was not returned by Sampler]"
            append_markdown(output_file, response)
        new_durable_additions = "none"
        try:
            local_summary = extract_section(response, "LOCAL_SUMMARY") or response
            chunk_summaries.append(
                bound_text(runtime, local_summary, limits.chunk_summary_tokens)
            )

            memory_delta = extract_section(response, "MEMORY_DELTA")
            if useful_memory_delta(memory_delta):
                bounded_delta = bound_text(
                    runtime,
                    memory_delta,
                    limits.memory_delta_tokens,
                )
                new_durable_additions = bounded_delta
                memory.text = append_memory_delta(
                    memory.text,
                    chunk.index,
                    bounded_delta,
                )
                memory_deltas.append(f"Chunk {chunk.index}: {bounded_delta}")
        except Exception as exc:
            if run_store is not None:
                run_store.chunk_failed(chunk, exc)
            raise
        image_notes.extend(
            note for note in image_summaries if note not in image_notes
        )

        append_markdown(
            output_file,
            "\n\n### New Durable Additions\n\n"
            f"{new_durable_additions}\n\n"
            "### State Size After Chunk\n\n"
            f"- Rolling memory tokens: {runtime.count_tokens(memory.text)}\n"
            f"- Memory deltas: {len(memory_deltas)}\n"
            f"- Image notes: {len(image_notes)}\n\n",
        )
        if run_store is not None:
            run_store.chunk_completed(
                chunk,
                chunk_summaries[-1],
                memory.text,
                memory_deltas=memory_deltas,
            )

    print("\n\n=== FINAL ABSTRACT ===", flush=True)
    if run_store is not None:
        image_notes = load_initial_image_notes(run_store)
    final_prompt = build_final_prompt_that_fits(
        runtime,
        memory,
        chunk_summaries,
        memory_deltas,
        image_notes,
        limits,
        output_file,
        run_store,
    )
    final_tokens = count_prompt_tokens(runtime, final_prompt)
    if run_store is not None:
        run_store.final_started(final_tokens)
    append_markdown(output_file, "## Final Pass\n\n")
    try:
        check_text_prompt_budget(runtime, final_prompt, "final abstract")
        with output_file.open("a", encoding="utf-8") as stream_file:
            final_abstract = runtime.chat(
                final_prompt,
                stream_file=stream_file,
                max_new_tokens=getattr(runtime, "final_output_tokens", None),
            )
    except Exception as exc:
        if run_store is not None:
            run_store.final_failed(exc)
        raise
    if not final_abstract:
        final_abstract = "[streamed final abstract was not returned by Sampler]"
        append_markdown(output_file, final_abstract)
    final_product_file = write_final_product_file(output_file, final_abstract)

    append_markdown(
        output_file,
        "\n\n## Final Rolling Memory\n\n"
        f"{memory.text}\n",
    )
    if run_store is not None:
        run_store.finish(final_abstract, final_product_file=final_product_file)

    return DigestionResult(
        chunk_summaries=chunk_summaries,
        final_abstract=final_abstract,
        rolling_memory=memory,
        image_notes=image_notes,
        memory_deltas=memory_deltas,
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
    memory_deltas: list[str],
    image_notes: list[str],
) -> str:
    summaries = "\n\n".join(
        f"CHUNK {index} SUMMARY:\n{summary}"
        for index, summary in enumerate(chunk_summaries, start=1)
    )
    deltas = "\n\n".join(memory_deltas) if memory_deltas else "none"
    images = "\n\n".join(image_notes) if image_notes else "none"
    return (
        "Write a detailed final paper digest from the accumulated digestion "
        "state. Stay faithful to the paper and avoid unsupported claims. The "
        "output should be substantially detailed: aim for 1200-1800 words if "
        "the supplied notes support it. Do not produce a short abstract unless "
        "there is too little source material.\n\n"
        "Final rolling memory:\n"
        f"{memory.text}\n\n"
        "Durable memory deltas, ordered:\n"
        f"{deltas}\n\n"
        "Ordered local summaries:\n"
        f"{summaries}\n\n"
        "Figure/table image notes, kept separate from memory:\n"
        f"{images}\n\n"
        "Return exactly these sections:\n"
        "DETAILED_DIGEST:\n"
        "KEY_CONTRIBUTIONS:\n"
        "METHOD_AND_ARCHITECTURE:\n"
        "TRAINING_AND_EVALUATION:\n"
        "RESULTS:\n"
        "FIGURES_AND_TABLES:\n"
        "LIMITATIONS_AND_FUTURE_WORK:\n"
        "SHORT_SUMMARY:\n"
    )


def digest_images_sequentially(
    runtime,
    chunk: ChunkPlan,
    images,
    output_file: Path,
    limits: DigestionLimits,
    run_store=None,
) -> list[str]:
    append_markdown(
        output_file,
        "### Image Prepass\n\n"
        f"{len(images)} images found. Reading them one at a time to avoid "
        "multi-image vision memory pressure.\n\n",
    )
    summaries: list[str] = []
    context = bound_text(runtime, format_chunk_text(chunk), limits.image_context_tokens)
    for index, image in enumerate(images, start=1):
        prompt = build_image_prompt(chunk, image.path.name, context)
        prompt_tokens = count_prompt_tokens(runtime, prompt)
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
                    images=image_arrays_or_none([image]),
                    stream_file=stream_file,
                )
        except Exception as exc:
            if run_store is not None:
                run_store.image_failed(chunk.index, index, image.path.name, exc)
            raise
        summary = extract_section(response or "", "IMAGE_SUMMARY") or response or ""
        summary = bound_text(runtime, summary.strip(), limits.image_summary_tokens)
        note = f"Chunk {chunk.index} image {index} ({image.path.name}): {summary}"
        summaries.append(note)
        if run_store is not None:
            run_store.image_completed(
                chunk.index,
                index,
                image.path.name,
                len(summary),
                note=note,
            )
        append_markdown(output_file, "\n\n")
    return summaries


def build_image_prompt(chunk: ChunkPlan, image_name: str, context: str) -> str:
    return (
        "Read this research-paper figure/table image in the context of the "
        "paper. Do not describe the image exhaustively. Explain what the image "
        "contributes to the paper: the claim, evidence, or conclusion a reader "
        "should take from it. Preserve only visual details needed for the final "
        "paper digest. If the image is unreadable, say so briefly.\n\n"
        f"Chunk {chunk.index}: {chunk.title()}\n"
        f"Image file: {image_name}\n\n"
        "Surrounding chunk text excerpt:\n"
        f"{context}\n\n"
        "Image payload:\n"
        "<|image|>\n\n"
        "Return exactly this section, in 60-100 words:\n"
        "IMAGE_SUMMARY:\n"
    )


def build_final_prompt_that_fits(
    runtime,
    memory: RollingMemory,
    chunk_summaries: list[str],
    memory_deltas: list[str],
    image_notes: list[str],
    limits: DigestionLimits,
    output_file: Path,
    run_store=None,
) -> str:
    memory_text = memory.text
    for summary_limit in (
        limits.final_summary_tokens,
        *limits.final_summary_fallback_tokens,
    ):
        bounded_summaries = [
            bound_text(runtime, summary, summary_limit)
            for summary in chunk_summaries
        ]
        bounded_deltas = bound_list_by_tokens(
            runtime,
            memory_deltas,
            limits.final_memory_delta_tokens,
        )
        bounded_images = bound_list_by_tokens(
            runtime,
            image_notes,
            limits.final_image_note_tokens,
        )
        prompt = build_final_prompt(
            RollingMemory(memory_text),
            bounded_summaries,
            bounded_deltas,
            bounded_images,
        )
        if count_prompt_tokens(runtime, prompt) <= runtime.safe_input_tokens:
            return prompt

    compacted_memory = compact_memory(
        runtime,
        memory_text,
        limits.emergency_memory_target_tokens,
        limits,
        output_file,
        run_store,
    )
    for summary_limit in limits.final_compacted_summary_tokens:
        prompt = build_final_prompt(
            RollingMemory(compacted_memory),
            [bound_text(runtime, summary, summary_limit) for summary in chunk_summaries],
            bound_list_by_tokens(
                runtime,
                memory_deltas,
                limits.final_compacted_memory_delta_tokens,
            ),
            bound_list_by_tokens(
                runtime,
                image_notes,
                limits.final_compacted_image_note_tokens,
            ),
        )
        if count_prompt_tokens(runtime, prompt) <= runtime.safe_input_tokens:
            return prompt
    raise ValueError(
        "Final digest prompt does not fit even after emergency compaction."
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
        "DETAILED_DIGEST:",
        "KEY_CONTRIBUTIONS:",
        "METHOD_AND_ARCHITECTURE:",
        "TRAINING_AND_EVALUATION:",
        "RESULTS:",
        "FIGURES_AND_TABLES:",
        "LIMITATIONS_AND_FUTURE_WORK:",
        "SHORT_SUMMARY:",
        "IMPORTANT_CLAIMS_RESULTS:",
        "LIMITATIONS_OR_UNCERTAINTIES:",
        "FIGURES_TABLES_USED:",
        "IMAGE_SUMMARY:",
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


def load_initial_memory_deltas(run_store, chunk_summaries: list[str]) -> list[str]:
    if run_store is None:
        return []
    deltas = run_store.state.get("memory_deltas")
    if isinstance(deltas, list):
        return list(deltas)
    chunks = run_store.state.get("chunks", [])[: len(chunk_summaries)]
    recovered = []
    for chunk in chunks:
        delta = chunk.get("memory_delta")
        if delta:
            recovered.append(f"Chunk {chunk['index']}: {delta}")
    return recovered


def load_initial_image_notes(run_store) -> list[str]:
    if run_store is None:
        return []
    notes = run_store.state.get("image_notes")
    if isinstance(notes, list):
        return list(notes)
    recovered = []
    for chunk in run_store.state.get("chunks", []):
        for image in chunk.get("image_prepass", []):
            note = image.get("note")
            if note:
                recovered.append(note)
    return recovered


def bound_list_by_tokens(runtime, items: list[str], token_limit: int) -> list[str]:
    selected: list[str] = []
    for item in items:
        candidate = selected + [item]
        if runtime.count_tokens("\n\n".join(candidate)) <= token_limit:
            selected.append(item)
            continue
        remaining = token_limit - runtime.count_tokens("\n\n".join(selected))
        if remaining > 40:
            selected.append(bound_text(runtime, item, remaining))
        break
    return selected


def compact_memory(
    runtime,
    memory_text: str,
    token_limit: int,
    limits: DigestionLimits,
    output_file: Path,
    run_store=None,
) -> str:
    before_tokens = runtime.count_tokens(memory_text)
    target_tokens = max(
        limits.memory_compaction_min_target_tokens,
        round(token_limit * 0.65),
    )
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
    image_summaries: list[str] | None = None,
) -> dict[str, int]:
    chunk_text = format_chunk_text(chunk)
    prompt_tokens = count_prompt_tokens(runtime, prompt)
    image_summary_text = "\n".join(image_summaries or [])
    return {
        "prompt_text_tokens": prompt_tokens,
        "chunk_text_tokens": runtime.count_tokens(chunk_text),
        "rolling_memory_tokens": runtime.count_tokens(memory.text),
        "image_summary_tokens": runtime.count_tokens(image_summary_text),
        "image_count": len(images),
        "skipped_image_count": len(skipped_images),
        "image_tokens": len(images) * IMAGE_TOKENS,
        "estimated_total_input_tokens": prompt_tokens + len(images) * IMAGE_TOKENS,
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
    tokens = count_prompt_tokens(runtime, prompt)
    if tokens > runtime.safe_input_tokens:
        raise ValueError(
            f"The {label} prompt has {tokens} tokens, over the safe limit "
            f"{runtime.safe_input_tokens}."
        )


def append_markdown(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(text)


def write_final_product_file(output_file: Path, final_abstract: str) -> Path:
    final_file = output_file.with_name(f"{output_file.stem}.final.md")
    final_file.write_text(
        "# Final Product\n\n"
        f"{final_abstract.strip()}\n",
        encoding="utf-8",
    )
    return final_file


def image_arrays_or_none(images) -> list[object] | None:
    if not images:
        return None
    return [image.array for image in images]


def count_prompt_tokens(runtime, prompt: str) -> int:
    if hasattr(runtime, "count_prompt_tokens"):
        return runtime.count_prompt_tokens(prompt)
    return runtime.count_tokens(prompt)
