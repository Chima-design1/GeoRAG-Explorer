"""
Geological map processing module for GeoRAG Explorer.

Supports:
- PDF text extraction
- text block and bounding-box extraction
- vector drawing statistics
- embedded image detection/extraction
- high-resolution page rendering
- tiled map rendering
- OCR for image-based maps
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pymupdf
import pytesseract
from PIL import Image


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TESSERACT_PATH = Path(
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

if TESSERACT_PATH.exists():
    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_PATH)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean extracted PDF/OCR text while preserving useful line structure."""
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines: List[str] = []

    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()

        if line:
            lines.append(line)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PDF map extraction
# ---------------------------------------------------------------------------

def extract_map_pdf(pdf_path: str | Path) -> Dict[str, Any]:
    """
    Extract structural information from a geological map PDF.

    This function does not run OCR automatically. It identifies whether
    a map is text/vector-rich or image-based and returns metadata useful
    for deciding how the map should be processed.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"Map PDF not found: {pdf_path}")

    document = pymupdf.open(pdf_path)

    pages: List[Dict[str, Any]] = []

    total_characters = 0
    total_drawings = 0
    total_images = 0
    total_labels = 0

    for page_index, page in enumerate(document):
        page_number = page_index + 1

        text = clean_text(page.get_text("text"))

        blocks = page.get_text("blocks")

        text_blocks: List[Dict[str, Any]] = []

        for block in blocks:
            if len(block) < 5:
                continue

            x0, y0, x1, y1 = block[:4]
            block_text = clean_text(str(block[4]))

            if not block_text:
                continue

            text_blocks.append(
                {
                    "text": block_text,
                    "bbox": [x0, y0, x1, y1],
                }
            )

        drawings = page.get_drawings()

        image_list = page.get_images(full=True)

        page_width = page.rect.width
        page_height = page.rect.height

        character_count = len(text)
        drawing_count = len(drawings)
        image_count = len(image_list)
        label_count = len(text_blocks)

        total_characters += character_count
        total_drawings += drawing_count
        total_images += image_count
        total_labels += label_count

        pages.append(
            {
                "page_number": page_number,
                "width": page_width,
                "height": page_height,
                "text": text,
                "character_count": character_count,
                "labels": text_blocks,
                "label_count": label_count,
                "drawing_count": drawing_count,
                "image_count": image_count,
            }
        )

    document.close()

    processing_type = (
        "image"
        if total_characters < 100 and total_labels == 0
        else "text_vector"
    )

    return {
        "document_id": pdf_path.stem,
        "filename": pdf_path.name,
        "path": str(pdf_path),
        "page_count": len(pages),
        "processing_type": processing_type,
        "total_characters": total_characters,
        "total_labels": total_labels,
        "total_drawings": total_drawings,
        "total_images": total_images,
        "pages": pages,
    }


# ---------------------------------------------------------------------------
# Embedded image extraction
# ---------------------------------------------------------------------------

def extract_embedded_images(
    pdf_path: str | Path,
    output_dir: str | Path,
) -> List[Dict[str, Any]]:
    """
    Extract embedded raster images from a PDF.

    Returns metadata for each extracted image.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    document = pymupdf.open(pdf_path)

    extracted: List[Dict[str, Any]] = []

    try:
        for page_index, page in enumerate(document):
            page_number = page_index + 1

            images = page.get_images(full=True)

            for image_index, image_info in enumerate(images, start=1):
                xref = image_info[0]

                try:
                    image_data = document.extract_image(xref)
                except Exception:
                    continue

                image_bytes = image_data.get("image")
                extension = image_data.get("ext", "png")

                if not image_bytes:
                    continue

                filename = (
                    f"{pdf_path.stem}"
                    f"_page_{page_number}"
                    f"_image_{image_index}.{extension}"
                )

                output_path = output_dir / filename
                output_path.write_bytes(image_bytes)

                extracted.append(
                    {
                        "page_number": page_number,
                        "image_number": image_index,
                        "xref": xref,
                        "width": image_data.get("width"),
                        "height": image_data.get("height"),
                        "extension": extension,
                        "path": str(output_path),
                    }
                )
    finally:
        document.close()

    return extracted


# ---------------------------------------------------------------------------
# High-resolution page rendering
# ---------------------------------------------------------------------------

def render_map_page(
    pdf_path: str | Path,
    output_path: str | Path,
    page_number: int = 1,
    scale: float = 3.0,
) -> Dict[str, Any]:
    """
    Render a PDF page as a high-resolution PNG.
    """
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if page_number < 1:
        raise ValueError("page_number must be >= 1")

    document = pymupdf.open(pdf_path)

    try:
        if page_number > len(document):
            raise ValueError(
                f"page_number {page_number} exceeds PDF page count "
                f"{len(document)}"
            )

        page = document[page_number - 1]

        matrix = pymupdf.Matrix(scale, scale)

        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        pixmap.save(str(output_path))

        return {
            "path": str(output_path),
            "page_number": page_number,
            "scale": scale,
            "width": pixmap.width,
            "height": pixmap.height,
        }

    finally:
        document.close()


# ---------------------------------------------------------------------------
# Tile rendering
# ---------------------------------------------------------------------------

def render_map_tiles(
    pdf_path: str | Path,
    output_dir: str | Path,
    page_number: int = 1,
    rows: int = 2,
    columns: int = 2,
    scale: float = 3.0,
) -> List[Dict[str, Any]]:
    """
    Render a PDF page at high resolution and divide it into tiles.

    Tiling improves OCR accuracy on geological maps with many small labels.
    """
    if rows < 1 or columns < 1:
        raise ValueError("rows and columns must be >= 1")

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_render = (
        output_dir
        / f"{pdf_path.stem}_page_{page_number}_full.png"
    )

    render_map_page(
        pdf_path=pdf_path,
        output_path=temp_render,
        page_number=page_number,
        scale=scale,
    )

    image = Image.open(temp_render)

    width, height = image.size

    tile_width = width // columns
    tile_height = height // rows

    results: List[Dict[str, Any]] = []

    for row in range(rows):
        for column in range(columns):

            left = column * tile_width
            upper = row * tile_height

            right = (
                width
                if column == columns - 1
                else (column + 1) * tile_width
            )

            lower = (
                height
                if row == rows - 1
                else (row + 1) * tile_height
            )

            tile = image.crop(
                (left, upper, right, lower)
            )

            tile_filename = (
                f"{pdf_path.stem}"
                f"_page_{page_number}"
                f"_tile_{row + 1}_{column + 1}.png"
            )

            tile_path = output_dir / tile_filename

            tile.save(
                tile_path,
                format="PNG",
            )

            results.append(
                {
                    "page_number": page_number,
                    "row": row + 1,
                    "column": column + 1,
                    "path": str(tile_path),
                    "width": tile.width,
                    "height": tile.height,
                }
            )

    image.close()

    return results


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def ocr_image(
    image_path: str | Path,
    psm: int = 11,
) -> str:
    """
    Run Tesseract OCR on an image.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(image_path)

    try:
        text = pytesseract.image_to_string(
            image,
            config=f"--psm {psm}",
        )
    finally:
        image.close()

    return clean_text(text)


def ocr_tiles(
    tile_paths: List[str | Path],
    psm: int = 11,
) -> Dict[str, str]:
    """
    OCR multiple tiles and return text keyed by tile filename.
    """
    results: Dict[str, str] = {}

    for tile_path in tile_paths:
        tile_path = Path(tile_path)

        try:
            results[tile_path.name] = ocr_image(
                tile_path,
                psm=psm,
            )
        except Exception as exc:
            results[tile_path.name] = (
                f"OCR ERROR: {type(exc).__name__}: {exc}"
            )

    return results


# ---------------------------------------------------------------------------
# Automatic image-map OCR
# ---------------------------------------------------------------------------

def process_image_map(
    pdf_path: str | Path,
    output_dir: str | Path,
    page_number: int = 1,
    rows: int = 2,
    columns: int = 2,
    scale: float = 3.0,
    psm: int = 11,
) -> Dict[str, Any]:
    """
    Complete OCR workflow for an image-based map.

    PDF
      -> high-resolution render
      -> tiles
      -> OCR
      -> combined map text
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    tile_dir = (
        output_dir
        / f"{pdf_path.stem}_tiles"
    )

    tiles = render_map_tiles(
        pdf_path=pdf_path,
        output_dir=tile_dir,
        page_number=page_number,
        rows=rows,
        columns=columns,
        scale=scale,
    )

    ocr_results = ocr_tiles(
        [tile["path"] for tile in tiles],
        psm=psm,
    )

    combined_sections: List[str] = []

    for tile in tiles:
        tile_name = Path(tile["path"]).name

        text = ocr_results.get(
            tile_name,
            "",
        )

        if text:
            combined_sections.append(
                f"[{tile_name}]\n{text}"
            )

    combined_text = "\n\n".join(
        combined_sections
    )

    return {
        "document_id": pdf_path.stem,
        "filename": pdf_path.name,
        "page_number": page_number,
        "processing_type": "image_ocr",
        "rows": rows,
        "columns": columns,
        "scale": scale,
        "tile_count": len(tiles),
        "tiles": tiles,
        "ocr_text": combined_text,
        "ocr_results": ocr_results,
    }


# ---------------------------------------------------------------------------
# Automatic map processor
# ---------------------------------------------------------------------------

def process_map(
    pdf_path: str | Path,
    output_dir: str | Path,
) -> Dict[str, Any]:
    """
    Automatically choose text/vector extraction or OCR.

    Text/vector-rich maps are processed using PDF text extraction.

    Image-based maps are processed through:
        high-resolution render -> tiles -> OCR
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    metadata = extract_map_pdf(pdf_path)

    if metadata["processing_type"] == "image":
        ocr_result = process_image_map(
            pdf_path=pdf_path,
            output_dir=output_dir,
        )

        metadata["ocr"] = ocr_result
        metadata["ocr_text"] = ocr_result["ocr_text"]

    else:
        metadata["ocr"] = None
        metadata["ocr_text"] = ""

    return metadata
