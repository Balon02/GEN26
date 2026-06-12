# GEN26 Generation Process

This document explains how GEN26 turns a LaTeX research paper into streamed
chunk digests, append-only rolling memory, persisted run state, and a final
presentable Markdown digest.

## Supported Entry Points

GEN26 currently has two active launch paths.

The interactive CLI path is:

```bash
uv run python main.py digest attention.tar.gz --output attention.md
```

On smaller GPUs, lower the runtime cache length:

```bash
uv run python main.py digest attention.tar.gz --output attention.md --max-tokens 8192
```

This path loads Gemma, parses the paper, opens the curses planner, lets the user
choose the chunking structure, then runs generation.

The resume path is:

```bash
uv run python main.py resume attention.md
```

Resume accepts the same runtime knob:

```bash
uv run python main.py resume attention.md --max-tokens 8192
```

This path loads the previous JSON run state associated with `attention.md`,
marks any chunk that was still `running` as interrupted, reparses the original
source, reapplies the saved node include/bundle/split state, opens the planner
again, and continues from the first chunk whose node signature no longer has a
completed matching result.

The non-interactive Python path is:

```python
from gen26 import digest_auto

digest_auto("attention.tar.gz", "attention.md", max_tokens=8192)
```

This path does not open the planner. It marks every included top-level child of
the paper root as `WHOLE`, validates that each top-level chunk fits the safe
text budget, prints the resulting chunk list, and runs the same digestion
engine used by the CLI.

## Runtime Initialization

Generation starts by constructing `GemmaDigestRuntime`.

The runtime module sets JAX memory environment variables before importing
Gemma/JAX-related packages:

```text
XLA_PYTHON_CLIENT_MEM_FRACTION=1.0
XLA_PYTHON_CLIENT_ALLOCATOR=vmm
```

These are intentionally placed at module import time in `gen26/gemma_runtime.py`
before `from gemma import gm`. The goal is to match the working Gemma/JAX memory
setup discovered during development.

`GemmaDigestRuntime.__init__()` then:

1. Loads `.env` from the repository root so Kaggle credentials and any local
   environment are available.
2. Downloads or resolves `google/gemma-3/flax/gemma3-4b-it` with `kagglehub`.
3. Builds a multimodal `gm.nn.Gemma3_4B(text_only=False)` model.
4. Loads model parameters with `gm.ckpts.load_params()`.
5. Loads `gm.text.Gemma3Tokenizer`.
6. Derives the usable input budget from the configured `max_tokens` cache
   length.
7. Reads the model's declared vision encoder size and requires square input.
8. Creates two low-level `gm.text.Sampler` instances:
   - `self.sampler` for normal chunk and image-pass generation, with
     `max_out_length=768`.
   - `self.final_sampler` for the final product, with `max_out_length=3072`.

The runtime deliberately uses `gm.text.Sampler` instead of `ChatSampler`. GEN26
manages prompt state explicitly and formats each request as a single Gemma
instruction prompt. There is no implicit multi-turn chat history inside the
Gemma sampler.

## Prompt Formatting

Every prompt is wrapped by `GemmaDigestRuntime.format_prompt()`:

```text
<start_of_turn>user
...prompt...
<end_of_turn>
<start_of_turn>model
```

The project-facing placeholder `<|image|>` is replaced with Gemma's
`<start_of_image>` token before sampling.

This means the text passed to the tokenizer and sampler is not just the raw
prompt body. Prompt budget checks use `runtime.count_prompt_tokens()`, which
counts the fully wrapped prompt. That is important because the control tokens
also consume context.

## Source Loading

The parser accepts three source shapes:

1. `.tar.gz` arXiv-style archive.
2. Directory containing `.tex` files.
3. A single `.tex` file.

For `.tar.gz`, `load_latex_source()` extracts the archive into a temporary
directory with `tarfile.extractall(..., filter="data")`, then later removes
that directory through `LoadedSource.cleanup()`.

For all source shapes, the parser finds a main `.tex` file using
`find_main_tex()`. It prefers files containing `\begin{document}` and, among
those, prefers `ms.tex`, shorter paths, and then lexicographic filename order.

The parser expands `\input{...}` and `\include{...}` recursively with
`read_with_inputs()`. Included paths are searched relative to the including file
first and then relative to the project root. A `seen` set prevents recursive
include loops. Missing includes become a textual marker rather than a hard
failure.

Comments are stripped by `strip_comments()` before include expansion. A percent
sign starts a comment unless it is escaped as `\%`.

## LaTeX Parsing

GEN26 uses `pylatexenc` for the structural parse. The central parser call is:

```python
LatexWalker(text).get_latex_nodes()
```

The parser does not try to fully compile LaTeX. It builds a useful semantic tree
for digestion:

- `paper`
- `metadata`
- `abstract`
- `part`
- `chapter`
- `section`
- `subsection`
- `subsubsection`
- `paragraph`
- mathematical and semantic environments such as `equation`, `align`,
  `figure`, `table`, `theorem`, `proof`, `definition`, `lemma`,
  `proposition`, `corollary`
- `bibliography`

The parser maintains a section stack. When it sees a section macro, it pops the
stack until it finds a lower-level parent, creates a new `PaperNode`, and pushes
that section node. Non-section prose is collected into paragraph buffers and
flushed when a section or block environment begins.

Block environments are attached under the current section parent. Their raw
LaTeX block is converted to text with `LatexNodes2Text`. Captions, labels,
references, and image paths are extracted from the raw block so later stages can
present useful model context and load visual inputs.

Bibliography environments are parsed but marked `EXCLUDE` by default. This lets
the tree preserve that the bibliography exists without spending model context on
reference lists.

Unsectioned top-level `paragraph` nodes are treated as front matter and excluded
when the document has real sections. Paragraphs inside sections remain included.

## Paper Tree Model

The parsed document is represented by `PaperNode`.

Each node stores:

- `order`: stable traversal ID assigned during parsing.
- `node_type`: semantic type such as `section`, `figure`, or `paragraph`.
- `title`: section title, generic block title, caption, label, or fallback.
- `text`: plain text content for leaves.
- `source_path`, `source_start`, `source_end`: source provenance.
- `labels` and `references`: extracted LaTeX labels and refs/cites.
- `caption`: figure/table caption when present.
- `image_paths`: resolved paths from `\includegraphics`.
- `token_count`: token count of the selectable content.
- `include_status`: `INCLUDE` or `EXCLUDE`.
- `digest_mode`: `AUTO`, `WHOLE`, or `SPLIT`.
- `children`: nested `PaperNode` objects.

Token counts are computed during parsing through a `TokenCounter` protocol. In
real runs, `RuntimeTokenCounter` delegates to the Gemma tokenizer inside
`GemmaDigestRuntime`, so planner numbers reflect the model that will actually
consume the prompts.

`recompute_parent_totals()` recursively updates parent token counts as the sum
of included children. It is called after parser defaults and after include or
exclude changes in the curses planner.

## Interactive Planning

The curses planner receives the parsed root node and a `TokenBudget`.

It displays a flattened view of the expanded tree. Each visible row knows:

- which `PaperNode` it represents,
- the tree depth,
- whether it is currently under a bundled ancestor,
- whether it is currently under an excluded ancestor.

The user can:

- move with arrows or `j`/`k`,
- expand and collapse nodes,
- include or exclude a selected subtree,
- bundle a selected node with `b`,
- split a selected node with `s`,
- reset the selected node to automatic planning with `a`,
- press Enter to accept the current plan.

Color/state meanings:

- Included automatic nodes are green.
- Excluded nodes and nodes under excluded ancestors are red/dimmed.
- Bundled nodes and descendants under a bundled ancestor are cyan.
- Explicit split nodes are yellow.
- Over-budget bundled nodes are highlighted red.
- The cursor row uses reverse video.

Every redraw validates the plan with `pack_chunks()`. If a node marked `WHOLE`
exceeds the chunk text limit, the planner reports an error and refuses to
continue until the user splits that node or changes the plan.

## Automatic Planning

`digest_auto()` uses `plan_top_level_chunks()`.

That function walks the direct children of the paper root. Every included child
is marked `DigestMode.WHOLE`. Excluded children, such as the default-excluded
bibliography or front matter, stay excluded.

It then calls `pack_chunks()` and performs one final explicit check that no
resulting chunk exceeds `budget.chunk_text_tokens`. If anything is too large, it
raises a `ValueError` before any model generation begins. The error lists the
chunk index, token count, and title of each over-budget chunk.

## Token Budget

The runtime has one public memory/context knob: `max_tokens`.

`max_tokens` maps directly to Gemma sampler `cache_length`. The default is
`10240`. This preserves the previous working configuration.

The usable input budget is derived from that cache length:

```text
max_output_tokens = 768
final_output_tokens = 3072
safe_input_tokens = min(7800, max_tokens - 2440)
```

The constant `2440` is the reserve implied by the original configuration:
`10240 cache length - 7800 usable input`.

For example:

```text
max_tokens=10240 -> safe_input_tokens=7800
max_tokens=8192  -> safe_input_tokens=5752
```

`TokenBudget` reserves room inside the safe input budget:

```text
chunk_text_tokens = usable_input_tokens
                  - rolling_memory_tokens
                  - instruction_tokens
```

With the runtime defaults, `usable_input_tokens=7800`,
`rolling_memory_tokens=900`, and `instruction_tokens=350`, the chunk text limit
is `6550`.

Prompt validation is stricter during actual generation than during tree packing.
Before sampling, GEN26 computes:

- full wrapped prompt text tokens,
- chunk text tokens,
- rolling memory tokens,
- image summary tokens,
- raw image count,
- skipped image count,
- image tokens,
- estimated total input tokens.

Each raw image is treated as `256` input tokens. A chunk is rejected if:

```text
prompt_text_tokens + image_count * 256 > runtime.safe_input_tokens
```

This is the final guard before a chunk reaches Gemma.

## Chunk Packing

`pack_chunks()` first calls `plan_digest_units()` to decide which nodes become
digest units. In the current implementation, each digest unit becomes one
`ChunkPlan`.

Planning rules:

- Excluded subtrees produce no units.
- The root `paper` node delegates to its children.
- A node marked `WHOLE` must fit the chunk text limit and then becomes one unit.
- A node marked `SPLIT` delegates to its children, unless it has no children.
- `AUTO` nodes digest at the default target level, currently `subsection`.
- Metadata and abstract nodes are level `0`, so they naturally become digest
  units when included.
- Leaves become digest units.
- Oversized automatic non-leaf nodes are split into children.

The important point is that `WHOLE` is a hard instruction: if that node is too
large, planning raises rather than silently splitting it. That is how the
planner and auto mode force the user or caller to handle too-large chunks
explicitly.

## Image Loading

Images are discovered during LaTeX parsing from `\includegraphics{...}` inside
figure/table/block environments.

When a chunk starts, `load_chunk_images()` traverses the selected nodes inside
that chunk and collects unique image paths. Excluded descendant nodes are
ignored.

Raster formats are loaded with OpenCV:

- `.png`
- `.jpg`
- `.jpeg`
- `.webp`
- `.gif`

PDF figures are rendered by invoking:

```text
pdftoppm -f 1 -singlefile -scale-to <image_size> -png <path> <output_prefix>
```

Only the first page is rendered. `image_size` comes from the Gemma model's
vision encoder configuration, currently expected to be `896`.

`pdftoppm` is a system executable from Poppler, not a Python package. On Colab
or Debian-like systems, install it with `apt-get install -y poppler-utils`.
If the executable is missing, GEN26 treats the PDF as a skipped image and
continues digesting the paper.

All images are converted to RGB, resized to fit inside an `image_size x
image_size` square while preserving aspect ratio, and pasted onto a white
square canvas. This avoids wasting resources on oversized rendering while still
matching the model's declared vision input resolution.

## Image Generation Strategy

The code distinguishes between raw images passed directly into a chunk prompt
and sequential image prepass notes.

`MAX_RAW_IMAGES_PER_CHUNK_CALL` is currently `1`.

If a chunk has zero or one image, that image is included directly in the main
chunk prompt. The prompt contains one `<|image|>` placeholder per image. The
runtime verifies that the number of formatted `<start_of_image>` placeholders
exactly equals the number of image arrays.

If a chunk has more than one image, GEN26 avoids a multi-image vision batch.
Instead, it runs `digest_images_sequentially()`:

1. The chunk text is bounded to a 900-token context excerpt.
2. Each image is processed alone with `build_image_prompt()`.
3. The image prompt asks what the figure contributes to the paper, not for an
   exhaustive visual description.
4. The model returns `IMAGE_SUMMARY:`.
5. The bounded image note is stored as:
   `Chunk <n> image <i> (<filename>): <summary>`.
6. These notes are fed into the later main chunk text prompt as "Sequential
   image readings".
7. The same notes are persisted separately and included in the final synthesis.

This design reduces OOM risk from multi-image calls while preserving visual
evidence for final output.

## Chunk Generation

For each chunk, `digest_chunks()` performs the same high-level sequence:

1. Print a console header with chunk number, token count, and node count.
2. Load chunk images.
3. Append a chunk heading to the output Markdown.
4. If needed, run sequential image prepass.
5. Build the main chunk prompt with:
   - current rolling memory,
   - formatted chunk source text,
   - raw image placeholders or image prepass notes,
   - skipped-image messages,
   - instructions for exact output sections.
6. Compute and persist prompt statistics.
7. Check input budget.
8. Stream Gemma output to the console and the Markdown file simultaneously.
9. Extract `LOCAL_SUMMARY`.
10. Extract `MEMORY_DELTA`.
11. Append only useful new durable memory deltas to rolling memory.
12. Persist chunk completion state.

The chunk prompt requires these sections:

```text
LOCAL_SUMMARY:
MEMORY_DELTA:
IMPORTANT_CLAIMS_RESULTS:
LIMITATIONS_OR_UNCERTAINTIES:
FIGURES_TABLES_USED:
```

`LOCAL_SUMMARY` is bounded and stored in `chunk_summaries`. It supports the
final synthesis but is not itself appended to rolling memory.

`MEMORY_DELTA` is the only chunk output that changes rolling memory. The model
is explicitly told not to rewrite the whole memory and to write `none` if there
are no new durable facts.

This keeps each stage focused on new information rather than repeatedly
rebuilding or restating the entire accumulated state.

## Rolling Memory

Rolling memory starts with fixed headings:

```text
Running abstract: empty
Key claims: empty
Methods and datasets: empty
Metrics and results: empty
Definitions and notation: empty
Limitations and caveats: empty
Unresolved ambiguities: empty
```

When a chunk produces a useful `MEMORY_DELTA`, the delta is appended:

```text
Chunk N durable additions:
...
```

The complete ordered delta list is also stored independently as
`memory_deltas`. This gives the final stage two views of accumulated knowledge:

- the current rolling memory text,
- the chronological list of local durable additions.

Routine compaction is intentionally avoided. Compaction only occurs in
`build_final_prompt_that_fits()` if the final synthesis prompt cannot fit after
progressively bounding summaries, deltas, and image notes.

Emergency compaction asks the model to preserve durable facts and remove
duplicated wording, parser artifacts, source-text copies, and stale local
details. The compaction result is itself streamed to the output Markdown and
logged in run state.

## Markdown Streaming

The model output is streamed token by token by `GemmaDigestRuntime.chat()`.

For every returned token:

1. End markers such as `<end_of_turn>` and `<eos>` are ignored.
2. Text is appended to an in-memory token list.
3. Text is written to stdout.
4. If a stream file was provided, text is written to that file and flushed.

This is why output appears live in the terminal and lands in the `.md` file at
the same time.

If the sampler stream completes but returns an empty assembled string, the code
adds a fallback marker to the Markdown file so the run output remains
structurally readable.

## Run State And Logs

Every digestion run has four output files:

```text
paper.md
paper.final.md
paper.json
paper.log.jsonl
```

The Markdown file receives streamed chunk output, prompt budget notes, image
prepass output, final pass output, and final rolling memory.

The `.final.md` file contains only the final product and is meant to be easier
to present.

The `.json` file is mutable structured run state. It stores:

- status,
- source path,
- output path,
- runtime metadata,
- token budget,
- plan version,
- node include/digest states,
- chunk records,
- completed summaries,
- memory deltas,
- image notes,
- rolling memory,
- last completed chunk,
- final prompt and output metadata.

The `.log.jsonl` file is append-only event logging. Events include:

- `run_started`
- `chunk_started`
- `chunk_completed`
- `chunk_failed`
- `image_started`
- `image_completed`
- `image_failed`
- `memory_compaction_started`
- `memory_compaction_completed`
- `final_started`
- `final_failed`
- `run_completed`

The JSON state supports resume. The JSONL log supports debugging by preserving a
time-ordered trace of prompt sizes, image processing, failures, and completion
events.

## Resume Mechanics

Resume starts from the Markdown output path. `RunStore` derives the state path
by changing the suffix to `.json`.

The resume sequence is:

1. Load JSON state.
2. Mark any chunks with status `running` as `interrupted`.
3. Load Gemma runtime.
4. Reparse the original source path stored in JSON.
5. Reapply saved node states with `apply_node_states()`.
6. Reopen the curses planner.
7. Build the new chunk plan.
8. Compare new chunk signatures with previously completed chunk signatures.
9. Preserve the longest completed prefix with identical node orders.
10. Truncate completed summaries, memory deltas, and image notes to that prefix.
11. Continue generation from the first incomplete or changed chunk.

This allows a failed overlarge chunk to be split on resume while preserving
completed work before that point.

The comparison uses `node_orders`, not titles or text. That makes resume stable
against display-label changes and focuses on whether the planned semantic units
are the same.

## Final Synthesis

After all chunks complete, GEN26 builds a final prompt from:

- final rolling memory,
- ordered durable memory deltas,
- ordered local summaries,
- image/table notes kept outside rolling memory.

The final prompt asks for these exact sections:

```text
DETAILED_DIGEST:
KEY_CONTRIBUTIONS:
METHOD_AND_ARCHITECTURE:
TRAINING_AND_EVALUATION:
RESULTS:
FIGURES_AND_TABLES:
LIMITATIONS_AND_FUTURE_WORK:
SHORT_SUMMARY:
```

The final pass uses `runtime.final_output_tokens` when available, currently
`3072`, so it can produce a substantially longer final document than a local
chunk pass.

`build_final_prompt_that_fits()` first tries to fit the final prompt by bounding
per-chunk summaries and total memory delta/image-note budgets. If that does not
fit, it performs emergency rolling memory compaction and tries again with
tighter summary/image/delta limits.

The final response is streamed to the main Markdown output and then copied into
`<stem>.final.md` with a `# Final Product` heading.

## Failure Behavior

GEN26 intentionally fails early for predictable budget problems:

- A `WHOLE` node that exceeds the chunk text limit raises during planning.
- A prompt whose estimated total input exceeds `runtime.safe_input_tokens`
  raises before sampling.
- A final prompt that cannot fit even after emergency compaction raises.
- A mismatch between image placeholders and supplied image arrays raises before
  calling the sampler.

When a failure occurs during a tracked run, `RunStore` records the failure in
JSON state and writes an event to the JSONL log. This is what makes later
inspection and resume possible.

## Important Design Choices

The current implementation is intentionally sequential. It does not parallelize
model calls because rolling memory depends on ordered prior chunks.

The model does not own conversation state. GEN26 controls the entire prompt
state explicitly for every request.

Rolling memory is append-first, not regenerate-every-step. Each chunk supplies
only new durable additions. This avoids repeated summaries and makes the
intermediate product easier to audit.

Images are handled as model inputs, not OCR text. Multi-image chunks are reduced
through sequential image notes to keep memory use predictable.

The final product is synthesized once from accumulated state rather than being
treated as merely the last chunk summary.
