"""
parser.py

PDF ingestion pipeline using Docling.

Extracts:
- Text blocks
- Tables
- Images (summarised using VLM)

Designed for multimodal RAG pipelines.
"""

import base64
import hashlib
import io
import time
from pathlib import Path
from typing import Any, Dict, List

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem, TableItem

from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config import IMAGE_CACHE_DIR, MAX_IMAGE_DIM
from src.models.vlm import summarise_image


# ============================================================
# Chunk Types
# ============================================================

TEXT_CHUNK = "text"
TABLE_CHUNK = "table"
IMAGE_CHUNK = "image"


# ============================================================
# Text Splitter
# ============================================================

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""],
)


# ============================================================
# Utilities
# ============================================================

def _make_chunk_id(source: str, index: int, chunk_type: str) -> str:
    raw = f"{source}::{chunk_type}::{index}"
    return hashlib.md5(raw.encode()).hexdigest()


def _build_docling_converter() -> DocumentConverter:
    """Create Docling converter with OCR + image extraction."""

    options = PdfPipelineOptions()

    options.do_ocr = True
    options.ocr_options = EasyOcrOptions(force_full_page_ocr=False)

    # IMPORTANT for VLM
    options.generate_picture_images = True
    options.images_scale = 2.0

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options)
        }
    )


# ============================================================
# MAIN PARSER
# ============================================================

def parse_pdf(file_path: str | Path) -> Dict[str, Any]:

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    print(f"\n📄 Parsing PDF: {file_path.name}")

    start_time = time.time()

    converter = _build_docling_converter()
    result = converter.convert(str(file_path))
    doc = result.document

    filename = file_path.name
    chunks: List[Dict] = []

    text_count = 0
    table_count = 0
    image_count = 0

    # ============================================================
    # 1️⃣ TEXT
    # ============================================================

    for i, text_item in enumerate(doc.texts):

        content = (text_item.text or "").strip()

        if len(content) < 20:
            continue

        page_no = text_item.prov[0].page_no if text_item.prov else 0

        sub_chunks = _SPLITTER.split_text(content)

        for j, sub in enumerate(sub_chunks):
            idx = i * 100 + j

            chunks.append(
                {
                    "text": sub,
                    "metadata": {
                        "source": filename,
                        "page": page_no,
                        "chunk_type": TEXT_CHUNK,
                        "chunk_index": idx,
                    },
                    "chunk_id": _make_chunk_id(filename, idx, TEXT_CHUNK),
                }
            )
            text_count += 1

    # ============================================================
    # 2️⃣ TABLES
    # ============================================================

    for i, table in enumerate(doc.tables):

        if not isinstance(table, TableItem):
            continue

        try:
            df = table.export_to_dataframe()
            table_text = df.to_markdown(index=False)
        except Exception:
            table_text = str(table)

        caption = getattr(table, "caption", "") or ""
        page_no = table.prov[0].page_no if table.prov else 0

        combined = f"TABLE CONTENT:\n{table_text}\nCaption: {caption}"

        for j, sub in enumerate(_SPLITTER.split_text(combined)):
            idx = i * 100 + j

            chunks.append(
                {
                    "text": sub,
                    "metadata": {
                        "source": filename,
                        "page": page_no,
                        "chunk_type": TABLE_CHUNK,
                        "caption": caption,
                        "chunk_index": idx,
                    },
                    "chunk_id": _make_chunk_id(filename, idx, TABLE_CHUNK),
                }
            )
            table_count += 1

    # ============================================================
    # 3️⃣ IMAGES → VLM
    # ============================================================

    image_cache = Path(IMAGE_CACHE_DIR)
    image_cache.mkdir(parents=True, exist_ok=True)

    for i, picture in enumerate(doc.pictures):

        if not isinstance(picture, PictureItem):
            continue

        try:
            print(f"🖼 Processing image {i}")

            img = picture.get_image(doc)

            if img is None:
                print("⚠️ No image extracted")
                continue

            # resize large images
            w, h = img.size
            if max(w, h) > MAX_IMAGE_DIM:
                scale = MAX_IMAGE_DIM / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)))

            # save debug image
            img_path = image_cache / f"{filename}_img_{i}.png"
            img.save(img_path)

            # convert → base64
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_b64 = base64.b64encode(buf.getvalue()).decode()

            print("🚀 Calling VLM summariser...")

            summary = summarise_image(
                image_b64,
                source_filename=filename,
            )

            print("✅ VLM returned summary")

            caption = getattr(picture, "caption", "") or ""
            page_no = picture.prov[0].page_no if picture.prov else 0

            chunks.append(
                {
                    "text": f"IMAGE DESCRIPTION:\n{summary}\nCaption: {caption}",
                    "metadata": {
                        "source": filename,
                        "page": page_no,
                        "chunk_type": IMAGE_CHUNK,
                        "caption": caption,
                        "image_path": str(img_path),
                        "chunk_index": i,
                    },
                    "chunk_id": _make_chunk_id(filename, i, IMAGE_CHUNK),
                }
            )

            image_count += 1

        except Exception as e:
            print(f"❌ Image {i} failed: {e}")
            continue

    # ============================================================
    # STATS
    # ============================================================

    elapsed = round(time.time() - start_time, 2)

    stats = {
        "filename": filename,
        "total_chunks": len(chunks),
        "text_chunks": text_count,
        "table_chunks": table_count,
        "image_chunks": image_count,
        "processing_time_seconds": elapsed,
    }

    print("✅ Parsing complete:", stats)

    return {
        "chunks": chunks,
        "stats": stats,
    }