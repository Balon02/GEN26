# GEN26

Local Gemma digestion for arXiv-style LaTeX papers.

GEN26 parses a `.tex` file, a LaTeX project directory, or an arXiv-style
`.tar.gz` archive into a token-aware paper tree. It then digests selected chunks
sequentially with Gemma 3 4B IT, carrying forward append-only rolling memory and
writing both streamed chunk output and a final presentable digest.

## CLI

Run the interactive curses planner:

```bash
uv run python main.py digest attention.tar.gz --output attention.md
```

On smaller GPUs, lower the Gemma sampler cache length with the single runtime
knob:

```bash
uv run python main.py digest attention.tar.gz --output attention.md --max-tokens 8192
uv run python main.py resume attention.md --max-tokens 8192
```

Resume an interrupted run from the Markdown output path:

```bash
uv run python main.py resume attention.md
```

The planner lets you walk the paper tree, include or exclude subtrees, and
choose whether a node is bundled into one prompt or split into children.

Planner keys:

```text
Up/Down       move cursor
Right         expand node
Left          collapse node, or move to parent
i             include selected node/subtree
x             exclude selected node/subtree
b             bundle selected node/subtree into one prompt
s             split selected node into children
a             reset selected node to auto
Enter         accept plan and continue
q             quit
```

## Python Auto Run

For notebook, Colab, or non-interactive use, import the automatic handle:

```python
from gen26 import digest_auto

digest_auto("attention.tar.gz", "attention.md", max_tokens=8192)
```

The automatic plan bundles each included top-level paper node into one chunk and
preserves parser defaults such as bibliography exclusion. It prints the complete
chunk list and token budget before generation starts. If any top-level chunk is
too large, it raises before streaming model output.

## Outputs

Each run writes these files next to the requested Markdown output:

```text
attention.md         streamed chunk outputs and final digest
attention.final.md   final digest only, suitable for presentation
attention.json       run state, plan, chunk statuses, memory, image notes
attention.log.jsonl  append-only event log with prompt/context token sizes
```

Resume marks any previously running chunk as interrupted, restores the planner
state, lets you split or exclude failed chunks, and continues from the first
incomplete chunk while preserving completed summaries, memory deltas, image
notes, and rolling memory.

## Runtime

The runtime uses low-level `gm.text.Sampler` directly and formats each request
as one explicit Gemma instruction prompt. JAX memory environment variables are
set before importing Gemma/JAX:

```text
XLA_PYTHON_CLIENT_MEM_FRACTION=1.0
XLA_PYTHON_CLIENT_ALLOCATOR=vmm
```

The default `max_tokens` is `10240`, which produces the previous safe input
budget of `7800` tokens. Lower values keep output, rolling-memory, instruction,
and image settings fixed while reducing only the usable input budget.

Rolling memory is append-first. Each chunk returns a `MEMORY_DELTA` containing
only durable new facts; the runner appends that delta to existing memory and
persists ordered deltas separately. Routine compaction is avoided. Emergency
compaction is only used when the final prompt would otherwise exceed the safe
input budget.

The final pass uses a larger generation budget than local chunk passes and asks
for a detailed structured digest rather than a short abstract.

## LaTeX And Images

The parser uses `pylatexenc` to build the section/environment structure. It
resolves `\input{...}` and `\include{...}`, records labels, references,
captions, figures, token counts, and image paths on a `PaperNode` tree.

Unsectioned top-level paragraphs are treated as front matter and excluded by
default when the paper has real sections. Normal paragraph prose inside sections
is still included. Bibliography environments are excluded by default.

Figures referenced by `\includegraphics{...}` are sent with the owning chunk,
including figures found inside bundled descendant nodes. Raster images are
loaded with OpenCV. PDF figures are rendered with `pdftoppm -scale-to 896`,
then padded to Gemma 3 4B's declared `896x896` image input without changing
aspect ratio. If a chunk owns multiple images, the runner reads them one at a
time in an image prepass and feeds the resulting visual notes into the main
chunk prompt, avoiding multi-image vision batches.

`pdftoppm` is a system executable from Poppler, not a Python package. On Colab,
install it with:

```bash
apt-get update && apt-get install -y poppler-utils
```

If `pdftoppm` is missing, PDF figures are skipped and the digest continues.

Image prepass prompts ask what the figure contributes to the paper, not for an
exhaustive visual description. Image notes are persisted separately and are
merged into the final synthesis.
