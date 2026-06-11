# GEN26

Local CLI experiments for turning arXiv-style LaTeX papers into token-aware
Gemma digestion chunks.

## Current CLI

Use the Gemma 3 tokenizer for real counts:

```bash
uv run python main.py ingest arXiv-1706.03762v7.tar.gz
uv run python main.py tree arXiv-1706.03762v7.tar.gz
uv run python main.py budget arXiv-1706.03762v7.tar.gz
uv run python main.py digest arXiv-1706.03762v7.tar.gz --output digestion.md
uv run python main.py resume digestion.md
```

For parser-only checks without loading the Gemma tokenizer:

```bash
uv run python main.py --tokenizer approx ingest arXiv-1706.03762v7.tar.gz
uv run python main.py --tokenizer approx tree arXiv-1706.03762v7.tar.gz
uv run python main.py --tokenizer approx budget arXiv-1706.03762v7.tar.gz
```

Supported source inputs are a `.tex` file, a directory containing `.tex` files,
or an arXiv-style `.tar.gz` archive. The parser resolves `\input{...}` and
`\include{...}` and uses `pylatexenc` to build the section/environment
structure. It records labels, references, captions, figures, token counts, and
image paths on a simple `PaperNode` tree.

## Streaming Digestion

`digest` follows the same Gemma 3 model and JAX setup as `smoke_test_g3.py`:
Gemma 3 4B IT, multimodal sampling, `cache_length=10240`, and the same JAX
memory environment settings. The runtime uses the low-level `gm.text.Sampler`
directly and formats each request as one explicit Gemma instruction prompt.

The command first opens an interactive planner where you walk the paper tree,
include or exclude subtrees, and choose whether nodes should be digested whole
or split into children. It then streams each local chunk summary to the console
and appends the same returned model output to a Markdown file:

```bash
uv run python main.py digest arXiv-1706.03762v7.tar.gz \
  --output digestion.md
```

The run also writes sidecar files next to the Markdown output:

```text
digestion.md         streamed summaries and final abstract
digestion.final.md   final digest only, suitable for presentation
digestion.json       run state, active plan, chunk statuses, memory, image notes
digestion.log.jsonl  append-only event log with prompt/context token sizes
```

If the process dies during a chunk, resume from the Markdown path:

```bash
uv run python main.py resume digestion.md
```

Resume marks any previously running chunk as interrupted, restores the planner
state, lets you split or exclude the failed chunk, and continues from the first
incomplete chunk while preserving completed summaries, memory deltas, image
notes, and rolling memory.

Rolling memory is append-first. Each chunk returns a `MEMORY_DELTA` containing
only durable new facts; the runner appends that delta to the existing memory and
persists the ordered deltas separately. Routine compaction is avoided. Emergency
compaction is only used when the final prompt would otherwise exceed the safe
input budget.

The final pass uses a larger generation budget than local chunk passes and asks
for a detailed structured digest rather than a short abstract.

Unsectioned top-level paragraphs are treated as front matter and excluded by
default when the paper has real sections. Normal paragraph prose inside sections
is still included.

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

Figures referenced by `\includegraphics{...}` are sent with the owning chunk,
including figures found inside bundled descendant nodes. Raster images are
loaded with OpenCV. PDF figures are rendered with `pdftoppm -scale-to 896`,
then padded to Gemma 3 4B's declared `896x896` image input without changing
aspect ratio. If a chunk owns multiple images, the runner reads them one at a
time in an image prepass and feeds the resulting visual notes into the main
chunk prompt, avoiding multi-image vision batches.
Image prepass prompts ask what the figure contributes to the paper, not for an
exhaustive visual description. Image notes are persisted separately and are
merged into the final synthesis.
