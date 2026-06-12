# GEN26 Code Reference

This document describes every remaining source file, class, function, and
method in the cleaned GEN26 codebase.

## Repository Root

### `main.py`

`main.py` is the executable entry point for the CLI.

#### `main`

The file imports `main` from `gen26.cli`.

#### `if __name__ == "__main__"`

When run as a script, it calls `gen26.cli.main()` and raises `SystemExit` with
that return code. This makes the CLI return integer process status correctly.

### `pyproject.toml`

Defines the Python project metadata and dependencies.

Important runtime dependencies:

- `gemma`: model, tokenizer, checkpoint loading, and sampler APIs.
- `jax[cuda12]`: CUDA-enabled JAX runtime used by Gemma.
- `kagglehub`: resolves/downloads Gemma model artifacts from Kaggle.
- `opencv-python`: image loading, color conversion, and resizing.
- `pylatexenc`: LaTeX parsing and LaTeX-to-text conversion.
- `python-dotenv`: loads `.env` for local credentials/configuration.
- `orbax-checkpoint` and `tensorstore`: checkpoint-related dependencies needed
  by the Gemma/JAX stack.

## Package Export

### `gen26/__init__.py`

Defines the public import surface of the package.

#### `digest_auto`

Imports `digest_auto` from `gen26.auto` and exposes it as:

```python
from gen26 import digest_auto
```

#### `__all__`

Declares `["digest_auto"]` as the package's intended public API.

## Automatic Runner

### `gen26/auto.py`

Contains the non-interactive Python entry point and automatic top-level
planning.

#### `RuntimeTokenCounter`

Adapter class used by the LaTeX parser. It gives the parser a simple
`count(text)` interface while delegating actual tokenization to the initialized
Gemma runtime.

##### `RuntimeTokenCounter.name`

Class attribute set to `"gemma3"`. It identifies the token counter type.

##### `RuntimeTokenCounter.__init__(self, runtime)`

Stores the runtime object. The runtime is expected to expose
`count_tokens(text)`.

##### `RuntimeTokenCounter.count(self, text)`

Returns `self.runtime.count_tokens(text)`. This means parse-time token counts
use the same tokenizer that generation will use.

#### `digest_auto(source, output, max_tokens=10240, context_scale=1.0)`

Runs the entire digestion process without opening the curses planner.

Detailed behavior:

1. Lazily imports `digest_chunks`, `GemmaDigestRuntime`, and `RunStore`.
   This avoids loading Gemma/JAX merely because `gen26.auto` was imported.
2. Converts `source` and `output` into `Path` objects.
3. Creates `GemmaDigestRuntime(max_tokens=max_tokens)`, which loads the model,
   tokenizer, and samplers.
4. Loads the LaTeX source with `load_latex_source()`.
5. Parses the source into a `PaperNode` tree using `RuntimeTokenCounter`.
6. Builds a scaled `TokenBudget` from runtime cache, safe input limits, and
   `context_scale`.
7. Calls `plan_top_level_chunks()` to mark root children as whole chunks.
8. Prints the formatted token budget and chunk list before generation.
9. Creates a `RunStore` next to the output Markdown file.
10. Persists the initial run state.
11. Calls `digest_chunks()` to perform chunk generation, final synthesis, and
    output writing.
12. Cleans up temporary extracted LaTeX source in a `finally` block.

It returns `DigestionResult`.

#### `plan_top_level_chunks(root, budget)`

Creates the automatic chunk plan.

For each direct child of the root:

- if the child is not excluded, set `digest_mode = DigestMode.WHOLE`;
- if the child is excluded, leave it excluded.

It then calls `pack_chunks()`. Because `WHOLE` nodes are hard boundaries,
`pack_chunks()` raises if any selected top-level node exceeds the chunk text
budget.

The function also performs an explicit over-budget scan of the returned chunks
and raises a detailed `ValueError` listing every too-large chunk. This makes
auto mode fail before model generation starts.

## CLI

### `gen26/cli.py`

Contains the command-line interface and the interactive/resume orchestration.

#### `build_parser()`

Builds the `argparse.ArgumentParser` for the CLI.

It defines two subcommands:

- `digest SOURCE --output OUTPUT`
- `resume OUTPUT`

Both commands accept `--max-tokens`, the runtime knob for lowering Gemma sampler
cache length on smaller GPUs or raising it on larger accelerators.

The parser has no debug tokenizer/tree/budget subcommands after cleanup.

#### `add_digest_args(parser)`

Adds the shared `--output` option for commands that create a Markdown digestion
file.

`--output` defaults to `digestion.md` and is parsed as a `Path`.

#### `add_runtime_args(parser)`

Adds `--max-tokens`.

The argparse default is `None`. New digest runs resolve that to `10240`.
Resume resolves it to the stored cache length from the previous run unless the
user explicitly passes a new value. The resolved value is passed into
`GemmaDigestRuntime` as the sampler cache length.

#### `main(argv=None)`

CLI dispatcher.

Detailed behavior:

1. Builds the parser.
2. Parses `argv`, or process arguments when `argv` is `None`.
3. Calls `run_digest(args)` for the `digest` command.
4. Calls `run_resume(args)` for the `resume` command.
5. Reports an argparse error for any unknown command.

Returns an integer process status.

#### `RuntimeTokenCounter`

CLI-local version of the parser token counter adapter. It is intentionally the
same idea as `gen26.auto.RuntimeTokenCounter`.

##### `RuntimeTokenCounter.name`

Class attribute set to `"gemma3"`.

##### `RuntimeTokenCounter.__init__(self, runtime)`

Stores the initialized `GemmaDigestRuntime`.

##### `RuntimeTokenCounter.count(self, text)`

Delegates to `runtime.count_tokens(text)`.

#### `run_digest(args)`

Runs a fresh interactive digestion.

Detailed behavior:

1. Lazily imports `digest_chunks` and `GemmaDigestRuntime`.
2. Initializes Gemma runtime with `args.max_tokens`.
3. Loads the requested LaTeX source.
4. Parses it into a token-counted paper tree.
5. Builds a `TokenBudget` from runtime limits.
6. Opens the curses planner through `terminal_plan(root, budget)`.
7. Prints the accepted budget/chunk report.
8. Creates initial run state with `RunStore`.
9. Calls `digest_chunks()` to perform generation.
10. Prints the Markdown output path.
11. Cleans up extracted source in a `finally` block.

Returns `0` on successful completion.

#### `run_resume(args)`

Resumes a previous digestion run from the Markdown output path.

Detailed behavior:

1. Imports `RollingMemory`, `digest_chunks`, and `GemmaDigestRuntime`.
2. Creates `RunStore(args.output)`.
3. Loads the associated `.json` state file.
4. Marks any `running` chunks as `interrupted`.
5. Initializes Gemma runtime with `args.max_tokens`.
6. Reopens the original source path stored in run state.
7. Reparses the source into a fresh tree.
8. Applies saved include and digest-mode node states with `apply_node_states()`.
9. Rebuilds the token budget.
10. Opens the curses planner again so failed chunks can be split/excluded.
11. Updates the stored plan and computes the completed prefix to preserve.
12. Restores rolling memory and completed summaries for that prefix.
13. Calls `digest_chunks()` on only the remaining chunks with
    `append_output=True`.
14. Prints the Markdown output path.
15. Cleans up extracted source in a `finally` block.

Returns `0` on successful completion.

## Chunk Planning

### `gen26/chunking.py`

Defines token budgets, chunk records, and the node-to-chunk planning algorithm.

#### `NODE_LEVELS`

Maps node types to planning levels:

- metadata and abstract are level `0`;
- section is level `1`;
- subsection is level `2`;
- subsubsection is level `3`;
- paragraphs, environments, figures, tables, equations, and bibliography are
  level `4`.

The default automatic planning target is `subsection`.

#### `TokenBudget`

Dataclass that stores high-level context allocation.

Fields:

- `cache_length`: configured sampler cache length.
- `usable_input_tokens`: conservative input token ceiling.
- `reserved_output_tokens`: local generation output allowance.
- `rolling_memory_tokens`: expected room for rolling memory.
- `instruction_tokens`: expected room for prompt instructions and wrappers.

##### `TokenBudget.chunk_text_tokens`

Computed property:

```text
usable_input_tokens - rolling_memory_tokens - instruction_tokens
```

This is the rough maximum allowed source text in a chunk before prompt assembly.

#### `ChunkPlan`

Dataclass representing one planned model chunk.

Fields:

- `index`: 1-based chunk index.
- `nodes`: paper nodes included in the chunk.
- `token_count`: combined token count of the chunk's planned node content.

##### `ChunkPlan.title(self)`

Returns a compact human-readable title for the chunk.

If there are no nodes, it returns `"(empty)"`.

If the first and last node labels are the same, it returns that label. Otherwise
it returns `"<first> -> <last>"`.

#### `pack_chunks(root, budget, default_level="subsection")`

Converts a paper tree into a list of `ChunkPlan` objects.

It first obtains digest units from `plan_digest_units()`. Each digest unit
currently becomes exactly one chunk. This keeps chunk planning simple and makes
planner decisions easy to inspect.

#### `plan_digest_units(root, budget, default_level="subsection")`

Generator that validates the chunk text limit and yields digest units.

It converts `default_level` into a numeric target level and calls
`walk_digest_units()`.

Raises `ValueError` if the budget leaves no room for chunk text.

#### `walk_digest_units(node, target_level, token_limit, excluded)`

Recursive core of chunk planning.

Rules:

- Excluded ancestors suppress the whole subtree.
- The `paper` root delegates to children.
- `DigestMode.WHOLE` yields the node if it fits, otherwise raises.
- `DigestMode.SPLIT` delegates to children, unless the node has no children.
- `DigestMode.AUTO` yields a node if it is at the target level, level `0`, or a
  leaf and it fits.
- Oversized automatic non-leaf nodes are split into children.

This is the function that enforces explicit over-budget failures for bundled
nodes.

#### `parse_default_level(level)`

Validates and converts a node type name such as `"subsection"` into its numeric
planning level.

Raises `ValueError` with the allowed keys if the level is unknown.

#### `format_budget_report(chunks, budget)`

Builds the text report printed before generation.

It includes:

- cache length,
- usable input,
- instruction reservation,
- rolling memory reservation,
- reserved output,
- chunk text limit,
- numbered chunk list with token count, node count, and title,
- `OVER` marker for chunks exceeding the chunk text limit.

## Paper Tree

### `gen26/paper_tree.py`

Defines the in-memory document model shared by parser, planner, chunker,
digester, and run store.

#### `IncludeStatus`

String enum for whether a node participates in generation.

Values:

- `INCLUDE = "include"`
- `EXCLUDE = "exclude"`

#### `DigestMode`

String enum for how the planner/chunker treats a node.

Values:

- `AUTO`: let the chunker decide based on node level and size.
- `WHOLE`: force the node/subtree into one chunk if it fits.
- `SPLIT`: force traversal into children.

#### `PaperNode`

Dataclass representing one semantic unit in the paper tree.

Fields:

- `order`: stable integer ID assigned during parsing.
- `node_type`: semantic type string.
- `title`: display title.
- `text`: plain text content for a leaf node.
- `source_path`: source `.tex` file path.
- `source_start`: start offset in source text when known.
- `source_end`: end offset in source text when known.
- `labels`: labels extracted from LaTeX.
- `references`: refs/cites extracted from LaTeX.
- `caption`: figure/table caption when present.
- `image_paths`: resolved image paths.
- `token_count`: token count used for planning and reporting.
- `include_status`: include/exclude state.
- `digest_mode`: auto/whole/split state.
- `children`: nested `PaperNode` objects.

##### `PaperNode.add_child(self, node)`

Appends a child node and returns it. Returning the node makes parser code more
convenient when constructing trees.

##### `PaperNode.walk(self)`

Depth-first traversal generator. Yields the current node first and then all
descendants.

##### `PaperNode.selectable_text(self)`

Builds the text used for token counting and model context.

It may include:

- node type and title,
- caption,
- image filenames,
- node text.

Empty parts are omitted. Parts are separated by blank lines.

##### `PaperNode.display_label(self, max_chars=96)`

Returns a single-line display label for planner rows, chunk titles, and run
state.

It prefers `title`, then `caption`, then the first 48 characters of text. It
normalizes whitespace and truncates long labels to `max_chars`, adding `...`.

#### `recompute_parent_totals(node)`

Recursively recomputes parent `token_count` values from included children.

If a leaf is excluded, it contributes `0`.

If a parent is excluded, it also returns `0` to its own parent.

The function returns the included token total for the subtree.

## LaTeX Parser

### `gen26/latex_parser.py`

Loads LaTeX sources and converts them into a token-counted `PaperNode` tree.

#### Constants

##### `SECTION_LEVELS`

Maps supported section macros to nesting levels.

##### `BLOCK_ENVS`

Set of environments promoted to separate tree nodes, including math blocks,
figures, tables, theorem-like environments, proof-like environments, and
`thebibliography`.

##### `INPUT_RE`

Regex for `\input{...}` and `\include{...}`. Used before `pylatexenc` parsing
to inline project files.

##### `CAPTION_RE`

Regex for captions inside raw block environment text.

##### `LABEL_RE`

Regex for `\label{...}`.

##### `REF_RE`

Regex for refs and cite-like commands. It supports `ref`, `eqref`, `autoref`,
`pageref`, `cite`, `citep`, and `citet`.

##### `GRAPHICS_RE`

Regex for `\includegraphics[...]{...}`.

##### `IMAGE_EXTENSIONS`

Supported resolved image suffixes.

#### `TokenCounter`

Typing protocol required by the parser.

Any token counter must expose:

- `name: str`
- `count(text: str) -> int`

The runtime adapters in `auto.py` and `cli.py` satisfy this protocol.

#### `LoadedSource`

Dataclass representing loaded and expanded LaTeX source.

Fields:

- `root_dir`: root directory for source and assets.
- `main_file`: selected main `.tex` file.
- `text`: source text after comment stripping and include expansion.
- `temp_dir`: temporary extraction directory for archive sources.

##### `LoadedSource.cleanup(self)`

Deletes the temporary directory when one exists. Directory and single-file
sources have no temporary directory and therefore no cleanup work.

#### `OrderCounter`

Small mutable counter for assigning stable node order IDs.

##### `OrderCounter.__init__(self)`

Initializes the counter at `0`.

##### `OrderCounter.next(self)`

Returns the current value and increments the counter.

#### `load_latex_source(path)`

Loads source from `.tar.gz`, directory, or `.tex`.

For `.tar.gz`, it extracts to a temp directory, finds the main `.tex`, expands
inputs, and returns `LoadedSource` with `temp_dir` set.

For directories and single `.tex` files, it finds or uses the main file and
returns `LoadedSource` without temp cleanup.

Raises `ValueError` for unsupported input paths.

#### `find_main_tex(root)`

Finds the most likely main `.tex` file.

It searches recursively for `*.tex`. Files containing `\begin{document}` are
preferred. If multiple candidates exist, it sorts with priority for `ms.tex`,
shorter path depth, and filename.

Raises `FileNotFoundError` if no `.tex` file exists.

#### `read_with_inputs(path, root, seen)`

Reads one `.tex` file, strips comments, and recursively replaces `\input` and
`\include` commands with file contents.

`seen` prevents infinite recursion.

If an included file cannot be found, the command is replaced with a textual
missing-input marker.

#### `strip_comments(text)`

Removes LaTeX comments line by line.

An unescaped `%` starts a comment. Escaped `\%` remains part of the line.

#### `parse_loaded_source(source, token_counter)`

Converts a `LoadedSource` into a `PaperNode` tree.

Detailed behavior:

1. Creates the root `paper` node.
2. Extracts document body between `\begin{document}` and `\end{document}`.
3. Extracts metadata from `\title`, `\author`, and `\date`.
4. Adds metadata as a leaf when present.
5. Extracts and adds abstract as a leaf when present.
6. Removes the abstract from the main content to avoid duplicate parsing.
7. Parses remaining content with `parse_blocks()`.
8. Excludes unsectioned top-level front matter if real sections exist.
9. Recomputes parent token totals.
10. Returns the root node.

#### `document_body(text)`

Returns the content inside `\begin{document}` and `\end{document}`. If no
document begin marker exists, returns the full text.

#### `exclude_unsectioned_front_matter(root)`

If the root has real section-like children, marks top-level paragraph children
as excluded. This keeps pre-section front matter out of generation while leaving
paragraphs inside sections intact.

#### `first_group(text, command)`

Finds the first required-brace argument for a LaTeX macro such as `title`,
`author`, or `date`. It supports an optional bracket argument before the brace.

Returns the captured value or `None`.

#### `parse_blocks(text, root, order, source, token_counter)`

Main structural parser.

It uses `LatexWalker(text).get_latex_nodes()` and walks top-level parsed nodes.

It maintains:

- a `section_stack` of `(level, PaperNode)` pairs,
- buffered paragraph parts,
- the starting source offset of the current paragraph buffer.

When it sees a section macro:

1. flushes pending paragraph text,
2. computes section level,
3. pops higher/equal section levels from the stack,
4. attaches a new section node under the current parent,
5. pushes the new section.

When it sees a block environment:

1. flushes pending paragraph text,
2. creates an environment node with `add_environment()`.

All other parsed nodes are accumulated as paragraph source.

At the end, pending paragraph text is flushed.

#### `is_section_node(node)`

Returns `True` when a parsed `pylatexenc` node is a macro whose name exists in
`SECTION_LEVELS`.

#### `is_block_environment(node)`

Returns `True` when a parsed node is a `LatexEnvironmentNode` whose environment
name is listed in `BLOCK_ENVS`.

#### `section_title_source(node)`

Returns the LaTeX source for the section title argument.

It walks the macro argument list in reverse and returns the last non-`None`
argument. This handles optional arguments by preferring the required/title
argument.

Falls back to the macro name if no argument is available.

#### `current_parent(stack)`

Returns the `PaperNode` at the top of the section stack.

#### `add_paragraphs(parent, order, text, source_path, offset, token_counter)`

Splits accumulated non-environment text into paragraphs by blank lines.

Each paragraph is converted to plain text with `latex_to_text()`. Very short
paragraphs are skipped. Remaining paragraphs are attached as `paragraph` leaf
nodes through `add_leaf()`.

#### `add_environment(...)`

Creates a `PaperNode` for a block environment.

It extracts:

- caption,
- labels,
- references,
- graphics paths.

It strips trailing `*` from starred environments. `thebibliography` is renamed
to `bibliography` and marked excluded by default.

The title is chosen as caption, first label, or environment name.

The raw block is converted to text and attached as the node's model-facing text.

#### `find_graphics_paths(raw_block, root_dir, source_path)`

Finds all `\includegraphics` paths in a raw block and resolves each to an
existing file when possible.

#### `resolve_image_path(raw_path, root_dir, source_path)`

Resolves one graphics path.

It first checks paths relative to the current source file and project root.
Then it falls back to recursively searching the root directory for files whose
name starts with the requested raw name and whose suffix is supported.

#### `resolve_candidate_image_path(candidate)`

Returns a candidate image path if it exists and has a supported suffix.

If the candidate has no suffix, tries every supported extension.

Returns `None` if nothing exists.

#### `add_leaf(...)`

Creates a leaf `PaperNode` with text, source provenance, labels, and references,
then calls `count_and_attach()`.

Returns the created node.

#### `count_and_attach(parent, node, token_counter)`

Counts the node's selectable text with the provided token counter, stores the
count on `node.token_count`, and appends the node to `parent.children`.

#### `latex_to_text(text)`

Converts LaTeX source to readable plain text using `LatexNodes2Text`.

Then it normalizes references, collapses excessive blank lines, normalizes
spaces, removes whitespace before newlines, and strips leading/trailing
whitespace.

#### `normalize_refs(text)`

Converts refs and citations into bracketed readable markers.

Example shape:

```text
\ref{foo} -> [ref: foo]
```

The command name is preserved so the model can distinguish refs and citations.

## Image Handling

### `gen26/images.py`

Loads and normalizes visual assets for Gemma.

#### `ChunkImage`

Dataclass holding:

- `path`: source image path,
- `array`: RGB image array prepared for model input.

#### `chunk_image_paths(chunk)`

Collects unique image paths from all selected nodes inside a chunk.

It traverses each chunk node through `selected_nodes()`, skips excluded
subtrees, and deduplicates paths while preserving first-seen order.

#### `selected_nodes(node)`

Generator over a node and its descendants, skipping an entire subtree if the
current node is excluded.

#### `load_chunk_images(chunk, image_size)`

Loads every image path associated with a chunk.

For each image:

1. Calls `read_image()`.
2. If loading fails, records a skipped-image message.
3. Fits the image to a square model input with `fit_image_to_square()`.
4. Wraps it in `ChunkImage`.

Returns `(loaded_images, skipped_messages)`.

#### `read_image(path, image_size)`

Loads one image.

For PDFs, delegates to `render_pdf_first_page()`.

For raster files, uses OpenCV `imread()` and converts BGR to RGB.

Returns `None` if OpenCV cannot read the file.

#### `render_pdf_first_page(path, image_size)`

Renders the first page of a PDF to PNG using `pdftoppm`.

It uses:

```text
pdftoppm -f 1 -singlefile -scale-to <image_size> -png ...
```

The rendered PNG is read with OpenCV and converted to RGB.

Returns `None` if `pdftoppm` is missing, `pdftoppm` fails, or OpenCV cannot
read the rendered PNG.

#### `fit_image_to_square(image, image_size, np)`

Resizes an RGB image to fit inside an `image_size x image_size` square while
preserving aspect ratio.

It creates a white square canvas and centers the resized image on it.

## Gemma Runtime

### `gen26/gemma_runtime.py`

Owns model loading, tokenizer access, prompt formatting, sampler calls, and
image shape validation.

#### Module-level environment setup

Sets JAX memory variables before importing Gemma:

```python
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "vmm"
```

#### `REPO_ROOT`

Repository root derived from the runtime file location.

#### `GEMMA_MODEL`

Model identifier:

```text
google/gemma-3/flax/gemma3-4b-it
```

#### `DEFAULT_CACHE_LENGTH`

Default Gemma sampler cache length, `10240`.

#### `DEFAULT_SAFE_INPUT_TOKENS`

Default usable prompt-input budget, `7800`.

#### `CACHE_TO_INPUT_RESERVE`

Fixed reserve between cache length and usable input budget. It is computed from
the default configuration as `10240 - 7800 = 2440`.

#### `MAX_OUTPUT_TOKENS`

Normal local generation output cap, `768`.

#### `FINAL_OUTPUT_TOKENS`

Final synthesis output cap, `3072`.

#### `safe_input_tokens_for_cache(cache_length)`

Derives usable input context from the single public cache-size knob.

It returns:

```text
cache_length - CACHE_TO_INPUT_RESERVE
```

Changing `max_tokens` therefore moves the usable input budget up or down while
preserving the fixed reserve implied by the original configuration. Output
limits and image token accounting remain separate. Fixed context allocations
can be scaled with `context_scale`.

Raises `ValueError` when the cache length leaves no usable input context.

#### `GemmaDigestRuntime`

Runtime facade used by the rest of the package.

##### `GemmaDigestRuntime.__init__(self, max_tokens=DEFAULT_CACHE_LENGTH)`

Initializes the model stack.

It:

- loads `.env`,
- downloads/resolves Gemma model artifacts,
- builds `Gemma3_4B(text_only=False)`,
- loads params,
- loads tokenizer,
- configures cache length, derived safe input budget, output sizes, and image
  size,
- verifies square vision input,
- creates the normal sampler,
- creates the larger final-output sampler.

Important attributes:

- `model`
- `params`
- `tokenizer`
- `cache_length`, set from `max_tokens`
- `safe_input_tokens`, derived from `max_tokens`
- `max_output_tokens`
- `final_output_tokens`
- `image_height`
- `image_width`
- `image_size`
- `sampler`
- `final_sampler`

##### `GemmaDigestRuntime.format_prompt(self, prompt)`

Replaces `<|image|>` with `<start_of_image>` and wraps the prompt in Gemma turn
tokens.

##### `GemmaDigestRuntime.count_tokens(self, text)`

Counts plain text with the Gemma tokenizer and `add_bos=True`.

Returns `0` for blank strings.

##### `GemmaDigestRuntime.count_prompt_tokens(self, prompt)`

Formats the prompt exactly as generation will see it, then tokenizes it with
`add_bos=True`. This is the budget-relevant count.

##### `GemmaDigestRuntime.chat(...)`

Streams one model response.

Parameters:

- `prompt`: unwrapped prompt body.
- `images`: optional image array or list of arrays.
- `stream_file`: optional file handle for Markdown streaming.
- `max_new_tokens`: optional per-call output token cap.

Detailed behavior:

1. Formats the prompt.
2. Normalizes image arrays with `normalize_images_for_sampler()`.
3. Counts images.
4. Validates image placeholder count.
5. Chooses `final_sampler` if requested output exceeds normal output size.
6. Calls `sampler.sample(..., stream=True)`.
7. Writes every streamed token to stdout.
8. Writes every streamed token to `stream_file` when provided.
9. Returns the accumulated text without Gemma end markers.

#### `normalize_images_for_sampler(images)`

Converts image input to a `uint8` NumPy array with shape `H,W,C` or `N,H,W,C`.

This is intentionally not batched as `B,N,H,W,C` because `gm.text.Sampler`
receives a plain string prompt and internally adds the batch dimension.

Raises `ValueError` if dimensions are not 3D/4D or channel count is not 3.

#### `count_sampler_images(images)`

Returns:

- `0` for `None`,
- `1` for `H,W,C`,
- `N` for `N,H,W,C`.

Raises on unexpected dimensions.

#### `validate_image_placeholders(prompt, image_count)`

Counts `<start_of_image>` tokens in the formatted prompt and verifies that the
count equals the supplied image count.

Raises `ValueError` on mismatch.

## Digestion Pipeline

### `gen26/digestion.py`

Contains the core sequential generation algorithm.

#### Constants

##### `IMAGE_TOKENS`

Estimated input-token cost per raw image. Currently `256`.

##### `MEMORY_DELTA_TOKEN_LIMIT`

Maximum retained tokens for one chunk's memory delta. Currently `260`.

##### `MAX_RAW_IMAGES_PER_CHUNK_CALL`

Maximum raw images allowed in a main chunk prompt. Currently `1`.

##### `FINAL_SUMMARY_TOKEN_LIMIT`

Initial token bound per local summary during final prompt construction.

##### `FINAL_IMAGE_NOTE_TOKEN_LIMIT`

Total token budget for image notes in the final prompt.

##### `FINAL_MEMORY_DELTA_TOKEN_LIMIT`

Total token budget for ordered memory deltas in the final prompt.

##### `EMERGENCY_MEMORY_TARGET_TOKENS`

Target budget used when final prompt construction needs emergency memory
compaction.

#### `RollingMemory`

Dataclass wrapping rolling memory text.

The default text has fixed headings for abstract, claims, methods, metrics,
definitions, limitations, and unresolved ambiguities.

#### `DigestionResult`

Dataclass returned by `digest_chunks()`.

Fields:

- `chunk_summaries`
- `final_abstract`
- `rolling_memory`
- `image_notes`
- `memory_deltas`

#### `digest_chunks(...)`

Main generation engine.

Parameters:

- `runtime`: object exposing Gemma methods and token counters.
- `chunks`: planned chunks.
- `output_file`: Markdown output path.
- `rolling_memory_token_limit`: retained for budget compatibility.
- `run_store`: optional persistent run state.
- `initial_memory`: resume memory.
- `initial_summaries`: resume summaries.
- `append_output`: whether to append a resume section instead of overwriting.
- `total_chunks`: display total when resuming partial chunks.

Detailed chunk loop:

1. Initializes memory, summaries, deltas, image notes, and output file.
2. For each chunk, prints a console header.
3. Loads images and skipped-image notices.
4. Writes chunk metadata to Markdown.
5. Runs sequential image prepass when a chunk has more than one image.
6. Builds the main chunk prompt.
7. Computes prompt component stats.
8. Marks chunk started in run state.
9. Writes prompt budget details to Markdown.
10. Checks prompt budget.
11. Streams model output to console and Markdown.
12. Extracts and bounds `LOCAL_SUMMARY`.
13. Extracts, bounds, and appends useful `MEMORY_DELTA`.
14. Adds image notes.
15. Writes new durable additions and state size to Markdown.
16. Marks chunk complete in run state.

Final stage:

1. Reloads image notes from run store if available.
2. Builds a final prompt that fits.
3. Marks final generation started.
4. Streams final output.
5. Writes fallback text if sampler returned no assembled response.
6. Writes `<stem>.final.md`.
7. Appends final rolling memory.
8. Marks the run complete.
9. Returns `DigestionResult`.

#### `build_chunk_prompt(chunk, memory, images, skipped_images, image_summaries=None)`

Builds the main prompt for a chunk.

It includes:

- task instructions,
- current rolling memory,
- formatted current chunk text,
- raw image placeholders,
- skipped-image messages,
- sequential image notes,
- instructions about paragraph parser labels,
- exact required response sections.

The prompt explicitly says `MEMORY_DELTA` must contain only new durable facts
and must not rewrite rolling memory.

#### `build_final_prompt(memory, chunk_summaries, memory_deltas, image_notes)`

Builds the final synthesis prompt.

It includes final memory, ordered memory deltas, ordered local summaries, and
image notes. It asks for a detailed 1200-1800 word digest when the notes support
that length and requires fixed output sections.

#### `digest_images_sequentially(runtime, chunk, images, output_file, run_store=None)`

Runs a one-image-at-a-time prepass for chunks with multiple images.

For each image:

1. Builds a bounded text context from the chunk.
2. Builds an image prompt.
3. Logs image start.
4. Writes image budget info to Markdown.
5. Checks prompt budget.
6. Streams model output with a single image.
7. Extracts `IMAGE_SUMMARY`.
8. Bounds it to 180 tokens.
9. Stores a chunk/image/filename note.
10. Logs image completion.

Returns a list of image notes.

#### `build_image_prompt(chunk, image_name, context)`

Builds the image prepass prompt.

It asks the model to explain what the figure/table contributes to the paper:
claim, evidence, or conclusion. It explicitly tells the model not to describe
the image exhaustively.

#### `build_final_prompt_that_fits(...)`

Builds a final prompt that stays within `runtime.safe_input_tokens`.

It first tries decreasing per-summary limits:

- `700`
- `500`
- `350`
- `220`

For each attempt, it also bounds memory deltas and image notes by total token
budget.

If none fit, it calls `compact_memory()` and tries stricter final prompt limits.

Raises `ValueError` if the final prompt cannot fit even after compaction.

#### `format_chunk_text(chunk)`

Formats all nodes in a chunk into model-facing text by calling
`format_node_subtree()` for each node.

#### `format_node_subtree(node)`

Formats a node and its included descendants.

If the node has no children, it emits the node heading and leaf model text.

If the node has children, it emits the parent heading and then each selected
leaf with its own heading and model text.

If a node has children but no selected leaves, it falls back to the node's own
selectable text.

#### `format_node_heading(node)`

Builds a heading like:

```text
[0004] SECTION - Introduction
```

Paragraph nodes with generic title `"paragraph"` return an empty heading to
avoid cluttering prompts with parser artifacts.

#### `model_text_for_node(node)`

Builds text for a node using caption, image filenames, and node text.

#### `selected_leaves(node)`

Generator over included leaf nodes in a subtree.

If the current node is excluded, the entire subtree is skipped.

#### `extract_section(text, section_name)`

Extracts a named section from model output.

It finds `<section_name>:` and stops at the earliest known next section marker.
Returns an empty string when the section is absent.

Known markers include chunk, image, memory compaction, and final-output
sections.

#### `bound_text(runtime, text, token_limit)`

Truncates text by removing lines from the end until it fits the token limit.

If the text already fits, returns it unchanged.

#### `useful_memory_delta(delta)`

Returns `False` for empty deltas and common no-op values:

- `none`
- `n/a`
- `no new durable facts`

Returns `True` otherwise.

#### `append_memory_delta(memory_text, chunk_index, delta)`

Appends one durable delta to rolling memory under:

```text
Chunk N durable additions:
```

Returns the updated memory text.

#### `load_initial_memory_deltas(run_store, chunk_summaries)`

Loads memory deltas for resumed runs.

It prefers `run_store.state["memory_deltas"]`. If absent, it reconstructs
deltas from completed chunk records up to the number of existing summaries.

Returns an empty list when no run store exists.

#### `load_initial_image_notes(run_store)`

Loads image notes for resumed runs.

It prefers `run_store.state["image_notes"]`. If absent, it reconstructs notes
from per-chunk image prepass records.

Returns an empty list when no run store exists.

#### `bound_list_by_tokens(runtime, items, token_limit)`

Selects list items in order until the joined text would exceed a token limit.

If there is meaningful remaining room for the next item, it appends a bounded
version of that item, then stops.

#### `compact_memory(runtime, memory_text, token_limit, output_file, run_store=None)`

Runs emergency rolling-memory compaction.

It:

1. Counts current memory tokens.
2. Computes a target token count.
3. Prints and writes compaction metadata.
4. Logs compaction start.
5. Builds a compaction prompt.
6. Checks prompt budget.
7. Streams compaction output.
8. Extracts `COMPACTED_ROLLING_MEMORY`.
9. Bounds the compacted result.
10. Logs compaction completion.
11. Writes compacted memory to Markdown.
12. Returns compacted memory or a bounded fallback.

#### `build_memory_compaction_prompt(memory_text, target_tokens)`

Builds the prompt for emergency memory compaction.

It instructs the model to preserve durable scientific facts and remove
duplicated wording, parser artifacts, copied source text, and stale local
details.

#### `prompt_component_stats(...)`

Computes prompt and input-budget statistics for a chunk.

Returns a dictionary containing:

- prompt text tokens,
- chunk text tokens,
- rolling memory tokens,
- image summary tokens,
- raw image count,
- skipped image count,
- image tokens,
- estimated total input tokens,
- safe input token limit.

#### `check_prompt_budget(runtime, prompt_stats, chunk)`

Raises `ValueError` if estimated total input tokens exceed
`runtime.safe_input_tokens`.

The error tells the user to split the node further in the planner.

#### `check_text_prompt_budget(runtime, prompt, label)`

Counts a text-only prompt after runtime formatting and raises if it exceeds
`runtime.safe_input_tokens`.

Used for image prepass, memory compaction, and final synthesis prompts.

#### `append_markdown(path, text)`

Appends text to a Markdown file with UTF-8 encoding.

#### `write_final_product_file(output_file, final_abstract)`

Writes `<output_stem>.final.md` containing:

```text
# Final Product

...
```

Returns the final file path.

#### `image_arrays_or_none(images)`

Converts a list of `ChunkImage` objects into a list of raw image arrays.

Returns `None` for empty input, which tells the runtime to make a text-only
sampler call.

#### `count_prompt_tokens(runtime, prompt)`

Counts prompt tokens through `runtime.count_prompt_tokens()` when available.

Falls back to `runtime.count_tokens(prompt)` for simpler test doubles.

## Run Store

### `gen26/run_store.py`

Persists mutable run state and append-only logs.

#### `now_iso()`

Returns the current UTC timestamp as an ISO-8601 string.

#### `RunStore`

Manages files derived from a Markdown output path.

##### `RunStore.__init__(self, output_file)`

Stores:

- `output_file`,
- `state_file = output_file.with_suffix(".json")`,
- `log_file = output_file.with_suffix(".log.jsonl")`,
- empty in-memory `state`.

##### `RunStore.create(self, source, runtime, budget, root, chunks)`

Creates initial run state and clears the JSONL log.

The state includes:

- run metadata,
- runtime metadata,
- budget,
- node states,
- chunk records,
- empty completed summaries,
- empty memory deltas,
- empty image notes,
- empty rolling memory,
- last completed chunk set to `0`.

It saves state and logs `run_started`.

##### `RunStore.load(self)`

Loads JSON state from disk into `self.state` and returns it.

##### `RunStore.save(self)`

Updates `updated_at` and writes pretty JSON state to disk.

##### `RunStore.log(self, event, **fields)`

Appends one JSON object to the `.log.jsonl` file.

Every log record includes `time`, `event`, and the supplied fields.

##### `RunStore.append_markdown(self, text)`

Appends text directly to the Markdown output file.

This method exists but the digestion pipeline mostly uses its own
`append_markdown()` helper.

##### `RunStore.mark_interrupted_chunks(self)`

Finds chunks whose status is `running`, marks them `interrupted`, stores an
error string, logs `chunk_interrupted`, and saves state if anything changed.

Used at resume start.

##### `RunStore.update_plan(self, root, chunks)`

Updates persisted plan state after the resume planner returns a new plan.

It:

1. Reads old chunks.
2. Finds completed old chunks.
3. Computes the longest prefix where old completed chunk node orders equal new
   chunk node orders.
4. Builds new chunk records.
5. Copies completion metadata for the preserved prefix.
6. Increments `plan_version`.
7. Replaces node states and chunk records.
8. Truncates completed summaries, memory deltas, and image notes to the
   preserved prefix.
9. Marks the run as `running`.
10. Saves and logs `plan_updated`.

Returns the preserved prefix length. The caller resumes at `prefix + 1`.

##### `RunStore.chunk_started(self, chunk, prompt_stats)`

Marks a chunk `running`, stores start timestamp and prompt stats, saves state,
and logs `chunk_started`.

##### `RunStore.chunk_completed(self, chunk, summary, rolling_memory, memory_deltas=None)`

Marks a chunk complete.

It stores:

- completion timestamp,
- bounded chunk summary,
- last completed chunk index,
- completed summaries list,
- memory deltas,
- per-chunk memory delta,
- current rolling memory.

Then it saves state and logs `chunk_completed`.

##### `RunStore.image_started(self, chunk_index, image_index, image_name, prompt_tokens)`

Adds a running image prepass record under the chunk and logs `image_started`.

The log includes prompt token count and fixed image token estimate.

##### `RunStore.image_completed(...)`

Finds the most recent matching image record and marks it complete.

It stores completion time, summary length, and the image note. It also adds the
note to top-level `image_notes` if it is not already present. Then it saves and
logs `image_completed`.

##### `RunStore.image_failed(...)`

Marks a matching image prepass record failed and stores error type/message.

Then it saves and logs `image_failed`.

##### `RunStore.chunk_failed(self, chunk, error)`

Marks a chunk failed, stores error type/message, sets run status to `failed`,
saves, and logs `chunk_failed`.

##### `RunStore.final_started(self, prompt_tokens)`

Stores final prompt token count, saves, and logs `final_started`.

##### `RunStore.memory_compaction_started(self, before_tokens, target_tokens)`

Marks memory compaction running, stores before/target token counts, saves, and
logs `memory_compaction_started`.

##### `RunStore.memory_compaction_completed(self, before_tokens, after_tokens)`

Marks memory compaction no longer running, stores after-token count, saves, and
logs `memory_compaction_completed`.

##### `RunStore.final_failed(self, error)`

Marks the run failed during final generation, stores error type/message, saves,
and logs `final_failed`.

##### `RunStore.finish(self, final_abstract, final_product_file=None)`

Marks the run complete, stores final output character count, stores final
product path when provided, saves, and logs `run_completed`.

##### `RunStore.chunk_record(self, index)`

Returns the persisted chunk record for the given 1-based chunk index.

Raises `ValueError` if no record exists.

#### `budget_to_dict(budget)`

Serializes `TokenBudget` into plain JSON-compatible fields, including computed
`chunk_text_tokens`.

#### `chunk_records(chunks)`

Serializes `ChunkPlan` objects into initial pending chunk records.

Each record stores:

- index,
- node order signature,
- node display titles,
- node types,
- token count,
- status.

#### `chunk_signature(chunk)`

Returns the list of node order IDs for a chunk.

This is the stable identity used for resume-prefix matching.

#### `last_delta_for_chunk(memory_deltas, chunk_index)`

Finds the most recent memory delta prefixed with `Chunk <index>:` and returns
the delta body without that prefix.

Returns an empty string when no match exists.

#### `memory_deltas_for_prefix(completed, prefix)`

Reconstructs the ordered memory delta list from completed chunk records up to a
prefix length.

#### `image_notes_for_prefix(chunk_records_, prefix)`

Reconstructs image notes from image prepass records in the preserved completed
prefix.

#### `node_states(root)`

Serializes every node's order, include status, and digest mode.

#### `apply_node_states(root, states)`

Applies serialized include/digest mode state to a freshly parsed tree by node
order.

Unknown node orders are ignored.

## Terminal Planner

### `gen26/terminal_planner.py`

Implements the curses tree UI used by `digest` and `resume`.

#### `Row`

Dataclass representing one visible planner row.

Fields:

- `node`: displayed `PaperNode`.
- `depth`: visual indentation depth.
- `bundled_by`: ancestor node that bundles this row, if any.
- `excluded_by`: ancestor node that excludes this row, if any.

#### `PlannerState`

Dataclass for mutable planner UI state.

Fields:

- `root`: paper root.
- `budget`: token budget.
- `expanded`: set of expanded node IDs.
- `cursor`: current row index.
- `message`: status/error message displayed in the UI.

#### `terminal_plan(root, budget)`

Creates initial planner state and enters curses with `curses.wrapper()`.

The root and its direct children start expanded.

Returns the accepted chunk list.

#### `run_planner(screen, state)`

Main curses event loop.

It:

1. Initializes cursor visibility, keypad mode, default colors, and color pairs.
2. Builds visible rows.
3. Draws the screen.
4. Reads one key.
5. Mutates cursor, expansion, include status, or digest mode.
6. On Enter, validates the plan and returns chunks if no errors exist.
7. On `q`, raises `KeyboardInterrupt`.

#### `init_colors()`

Initializes curses color pairs:

- green for included/auto,
- red for excluded,
- cyan for bundled,
- yellow for split,
- blue background for an unused pair,
- red background for hard errors.

#### `visible_rows(root, expanded)`

Builds the flat list of visible planner rows from the tree.

It propagates `bundled_by` and `excluded_by` context to descendants so the UI
can show when a child is effectively bundled or excluded by an ancestor.

Collapsed nodes do not show descendants.

#### `draw(screen, state, rows)`

Renders the planner.

It:

- clears the screen,
- computes terminal dimensions,
- validates the current plan,
- draws title and keybinds,
- draws chunk/error/budget summary,
- draws current message,
- scrolls rows around the cursor,
- formats and colors each row,
- refreshes the screen.

#### `format_row(row, expanded, over_budget, is_cursor=False)`

Returns the display text for one tree row.

The row includes cursor marker, expand/collapse marker, include status,
digest mode, node order, node type, token count, display label, and `OVER` when
the row is over budget.

#### `display_status(row)`

Returns the displayed include status.

If the row is excluded by an ancestor, returns `exc`; otherwise returns the
current node's include status prefix.

#### `display_mode(row)`

Returns the displayed digest mode.

If the row is inside a bundled ancestor, returns `in-bndl`. If the row itself
is `WHOLE`, returns `bundle`. Otherwise returns the node's digest mode value.

#### `row_attr(row, is_cursor, over_budget)`

Returns curses attributes for a row based on error/excluded/bundled/split/auto
state and cursor status.

#### `add_line(screen, y, x, text, attr=0)`

Safely writes one line to the curses screen if `y` is visible. It truncates text
to fit the terminal width.

#### `current(rows, state)`

Returns the `PaperNode` at the current cursor row.

#### `expand_current(state, rows)`

Adds the current node order to the expanded set when it has children.

#### `collapse_or_parent(state, rows)`

If the current node is expanded, collapses it. Otherwise moves the cursor to the
nearest visible parent row.

#### `validate_plan(root, budget)`

Calls `pack_chunks()` and converts planning exceptions into UI error node IDs.

Returns `(chunks, errors)`.

If planning succeeds, it also flags chunks whose token count exceeds the chunk
text budget.

#### `over_budget_nodes_from_error(root, message)`

Best-effort parser for chunking error messages. It searches for a node order ID
inside the exception message and returns that node ID as the error set.

## Current File Roles

The remaining source files have these responsibilities:

```text
main.py                    CLI executable wrapper
gen26/__init__.py          public Python API export
gen26/auto.py              non-interactive automatic runner
gen26/cli.py               CLI orchestration and resume flow
gen26/chunking.py          token budgets and chunk planning
gen26/digestion.py         sequential generation pipeline
gen26/gemma_runtime.py     Gemma/JAX runtime and sampler wrapper
gen26/images.py            image/PDF loading and normalization
gen26/latex_parser.py      LaTeX loading and PaperNode tree construction
gen26/paper_tree.py        document tree data model
gen26/run_store.py         JSON state and JSONL event log persistence
gen26/terminal_planner.py  curses planning UI
```
