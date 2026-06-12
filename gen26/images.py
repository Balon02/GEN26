from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from gen26.chunking import ChunkPlan
from gen26.paper_tree import IncludeStatus, PaperNode


@dataclass
class ChunkImage:
    path: Path
    array: object


def chunk_image_paths(chunk: ChunkPlan) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for node in chunk.nodes:
        for image_node in selected_nodes(node):
            for path in image_node.image_paths:
                if path in seen:
                    continue
                seen.add(path)
                paths.append(path)
    return paths


def selected_nodes(node: PaperNode):
    if node.include_status == IncludeStatus.EXCLUDE:
        return
    yield node
    for child in node.children:
        yield from selected_nodes(child)


def load_chunk_images(
    chunk: ChunkPlan,
    image_size: int,
) -> tuple[list[ChunkImage], list[str]]:
    import cv2
    import numpy as np

    loaded: list[ChunkImage] = []
    skipped: list[str] = []
    for path in chunk_image_paths(chunk):
        image = read_image(path, image_size)
        if image is None:
            skipped.append(f"{path.name}: image could not be rendered or read")
            continue
        image = fit_image_to_square(image, image_size, np)
        loaded.append(ChunkImage(path=path, array=image))
    return loaded, skipped


def read_image(path: Path, image_size: int):
    import cv2

    if path.suffix.lower() == ".pdf":
        return render_pdf_first_page(path, image_size)

    image = cv2.imread(str(path))
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def render_pdf_first_page(path: Path, image_size: int):
    import cv2

    with tempfile.TemporaryDirectory(prefix="gen26-pdf-") as temp_dir:
        output_prefix = Path(temp_dir) / "page"
        try:
            result = subprocess.run(
                [
                    "pdftoppm",
                    "-f",
                    "1",
                    "-singlefile",
                    "-scale-to",
                    str(image_size),
                    "-png",
                    str(path),
                    str(output_prefix),
                ],
                check=False,
                text=True,
                capture_output=True,
            )
        except FileNotFoundError:
            return None
        if result.returncode != 0:
            return None
        rendered = output_prefix.with_suffix(".png")
        image = cv2.imread(str(rendered))
        if image is None:
            return None
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def fit_image_to_square(image, image_size: int, np):
    import cv2

    height, width = image.shape[:2]
    scale = min(image_size / width, image_size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_CUBIC,
    )
    canvas = np.full((image_size, image_size, 3), 255, dtype=resized.dtype)
    top = (image_size - resized_height) // 2
    left = (image_size - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas
