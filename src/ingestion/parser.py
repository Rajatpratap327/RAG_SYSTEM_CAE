"""
PDF ingestion pipeline using Docling.
Extracts text blocks, tables, and images as distinct chunk types,
enabling modality-aware retrieval in the RAG pipeline.
"""

import base64
import hashlib
import time
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    EasyOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem

from src.config import IMAGE_CACHE_DIR, MAX_IMAGE_DIM
from src.models.vlm import summarise_image


# ── Chunk type literals ──────────────────────────────────────────────────────
TEXT_CHUNK = "text"
TABLE_CHUNK = "table"
IMAGE_CHUNK = "image"


def _make_chunk_id(source: str, index: int, chunk_type: str) -> str:
    """Generate a deterministic chunk ID from source filename, index, and type."""
    raw = f"{source}::{chunk_type}::{index}"
    return hashlib.md5(raw.encode()).hexdigest()


def _build_docling_converter() -> DocumentConverter:
    """Configure and return a Docling DocumentConverter with OCR and image export."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = EasyOcrOptions(force_full_page_ocr=False)
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 2.0  # Higher resolution for VLM input

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def parse_pdf(file_path: str | Path) -> dict[str, Any]:
    """
    Parse a PDF file and extract text, table, and image chunks.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Dictionary with keys:
            - 'chunks': list of chunk dicts (text, metadata)
            - 'stats': ingestion summary statistics
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    t0 = time.time()
    converter = _build_docling_converter()
    result = converter.convert(str(file_path))
    doc = result.document
    filename = file_path.name

    chunks: list[dict[str, Any]] = []
    text_count = table_count = image_count = 0

    # ── 1. Text chunks ────────────────────────────────────────────────────
    for i, text_item in enumerate(doc.texts):
        content = text_item.text.strip()
        if not content or len(content) < 20:  # Skip trivially short fragments
            continue
        page_no = (
            text_item.prov[0].page_no if text_item.prov else 0
        )
        chunks.append(
            {
                "text": content,
                "metadata": {
                    "source": filename,
                    "page": page_no,
                    "chunk_type": TEXT_CHUNK,
                    "chunk_index": i,
                },
                "chunk_id": _make_chunk_id(filename, i, TEXT_CHUNK),
            }
        )
        text_count += 1

    # ── 2. Table chunks ───────────────────────────────────────────────────
    for i, item in enumerate(doc.tables):
        if not isinstance(item, TableItem):
            continue
        try:
            df = item.export_to_dataframe()
            table_text = df.to_markdown(index=False)
        except Exception:
            table_text = str(item)

        page_no = item.prov[0].page_no if item.prov else 0
        caption = getattr(item, "caption", "") or ""

        chunks.append(
            {
                "text": f"TABLE CONTENT:\n{table_text}\nCaption: {caption}",
                "metadata": {
                    "source": filename,
                    "page": page_no,
                    "chunk_type": TABLE_CHUNK,
                    "chunk_index": i,
                    "caption": caption,
                },
                "chunk_id": _make_chunk_id(filename, i, TABLE_CHUNK),
            }
        )
        table_count += 1

    # ── 3. Image chunks (VLM summarisation) ──────────────────────────────
    image_cache = Path(IMAGE_CACHE_DIR)
    image_cache.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(doc.pictures):
        if not isinstance(item, PictureItem):
            continue
        try:
            # Export image as PNG bytes
            img = item.get_image(doc)
            if img is None:
                continue

            # Resize if oversized
            w, h = img.size
            if max(w, h) > MAX_IMAGE_DIM:
                scale = MAX_IMAGE_DIM / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)))

            # Save to cache for debugging / auditing
            img_path = image_cache / f"{filename}_img_{i}.png"
            img.save(img_path)

            # Convert to base64 for VLM
            import io
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            # Ask VLM to describe the image in CAE context
            summary = summarise_image(b64, source_filename=filename)

            page_no = item.prov[0].page_no if item.prov else 0
            caption = getattr(item, "caption", "") or ""

            chunks.append(
                {
                    "text": f"IMAGE DESCRIPTION:\n{summary}\nCaption: {caption}",
                    "metadata": {
                        "source": filename,
                        "page": page_no,
                        "chunk_type": IMAGE_CHUNK,
                        "chunk_index": i,
                        "caption": caption,
                        "image_path": str(img_path),
                    },
                    "chunk_id": _make_chunk_id(filename, i, IMAGE_CHUNK),
                }
            )
            image_count += 1

        except Exception as exc:
            # Non-fatal: log and continue with remaining images
            print(f"⚠️  Skipping image {i} in {filename}: {exc}")
            continue

    elapsed = round(time.time() - t0, 2)
    stats = {
        "filename": filename,
        "total_chunks": len(chunks),
        "text_chunks": text_count,
        "table_chunks": table_count,
        "image_chunks": image_count,
        "processing_time_seconds": elapsed,
    }

    return {"chunks": chunks, "stats": stats}
