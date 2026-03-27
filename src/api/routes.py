"""
FastAPI route definitions.

Endpoints:
  GET  /health     — System status, model readiness, index statistics
  POST /ingest     — Upload and index a multimodal PDF
  POST /query      — Natural language query with RAG answer generation
  GET  /documents  — List all indexed documents
  DELETE /documents/{filename} — Remove a document from the index
"""

import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status

from src.api.schemas import (
    DeleteResponse,
    DocumentListResponse,
    HealthResponse,
    IngestionResponse,
    QueryRequest,
    QueryResponse,
    SourceReference,
)
from src.config import EMBED_MODEL, LLM_MODEL, VLM_MODEL
from src.ingestion.parser import parse_pdf
from src.retrieval.rag_chain import run_rag_query
from src.retrieval.vector_store import (
    add_chunks,
    delete_document,
    get_store_stats,
)

router = APIRouter()

# Temporary upload directory
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ── GET /health ──────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health and index statistics",
    tags=["System"],
)
async def health(request: Request) -> HealthResponse:
    """
    Returns system status including model configuration, vector index
    statistics, and server uptime.

    This endpoint does not invoke any models — it is safe to poll frequently.
    """
    stats = get_store_stats()
    uptime = round(time.time() - request.app.state.start_time, 1)

    return HealthResponse(
        status="ok",
        llm_model=LLM_MODEL,
        vlm_model=VLM_MODEL,
        embed_model=EMBED_MODEL,
        total_chunks=stats["total_chunks"],
        unique_documents=stats["unique_documents"],
        chunk_type_breakdown=stats["chunk_type_breakdown"],
        indexed_files=stats["indexed_files"],
        uptime_seconds=uptime,
    )


# ── POST /ingest ─────────────────────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a multimodal PDF into the vector index",
    tags=["Ingestion"],
)
async def ingest(
    file: Annotated[UploadFile, File(description="Multimodal PDF to ingest")],
) -> IngestionResponse:
    """
    Upload a PDF document, extract text/table/image chunks, generate
    VLM summaries for images, and embed all chunks into ChromaDB.

    **Processing pipeline:**
    1. Save uploaded file to disk.
    2. Parse with Docling → text blocks, tables, extracted images.
    3. Images → LLaVA VLM → text summaries.
    4. All chunks embedded with nomic-embed-text and stored in ChromaDB.

    Returns an ingestion summary with chunk counts per modality and
    total processing time.
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported. Please upload a .pdf file.",
        )

    # Persist uploaded file
    save_path = UPLOAD_DIR / file.filename
    try:
        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {exc}",
        ) from exc

    # Parse PDF
    try:
        result = parse_pdf(save_path)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"PDF parsing failed: {exc}",
        ) from exc

    # Index chunks
    if result["chunks"]:
        add_chunks(result["chunks"])

    s = result["stats"]
    return IngestionResponse(
        message=f"Successfully ingested '{file.filename}'",
        filename=file.filename,
        total_chunks=s["total_chunks"],
        text_chunks=s["text_chunks"],
        table_chunks=s["table_chunks"],
        image_chunks=s["image_chunks"],
        processing_time_seconds=s["processing_time_seconds"],
    )


# ── POST /query ──────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the indexed documents using RAG",
    tags=["Query"],
)
async def query(body: QueryRequest) -> QueryResponse:
    """
    Accept a natural language question, retrieve the most relevant
    chunks from ChromaDB, and generate a grounded answer using
    Mistral-Nemo.

    **Retrieval strategy:**
    - Default: hybrid search across all chunk types (text + table + image).
    - Optional `chunk_type_filter` restricts retrieval to a specific modality.

    **Answer grounding:**
    - The LLM is instructed to answer only from retrieved context.
    - Source references (filename, page, chunk type) are returned alongside
      the answer.
    """
    # Validate optional filter value
    if body.chunk_type_filter and body.chunk_type_filter not in (
        "text", "table", "image"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="chunk_type_filter must be one of: 'text', 'table', 'image'",
        )

    try:
        result = run_rag_query(
            question=body.question,
            top_k=body.top_k,
            chunk_type_filter=body.chunk_type_filter,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG pipeline error: {exc}",
        ) from exc

    return QueryResponse(
        question=body.question,
        answer=result["answer"],
        sources=[SourceReference(**s) for s in result["sources"]],
        retrieved_chunks=result["retrieved_chunks"],
    )


# ── GET /documents ───────────────────────────────────────────────────────────

@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all indexed documents",
    tags=["System"],
)
async def list_documents() -> DocumentListResponse:
    """
    Return the list of filenames currently indexed in ChromaDB.
    Useful for confirming which PDFs have been ingested.
    """
    stats = get_store_stats()
    files = stats["indexed_files"]
    return DocumentListResponse(
        indexed_files=files,
        total_documents=len(files),
    )


# ── DELETE /documents/{filename} ─────────────────────────────────────────────

@router.delete(
    "/documents/{filename}",
    response_model=DeleteResponse,
    summary="Remove a document from the vector index",
    tags=["System"],
)
async def remove_document(filename: str) -> DeleteResponse:
    """
    Delete all chunks belonging to the specified source document from
    the ChromaDB index.

    This is useful for updating the index when a document is revised —
    delete the old version, then re-ingest the updated PDF.
    """
    deleted = delete_document(filename)
    if deleted == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No chunks found for document '{filename}'. "
                   "Verify the filename matches exactly.",
        )
    return DeleteResponse(
        message=f"Successfully removed '{filename}' from the index.",
        filename=filename,
        chunks_deleted=deleted,
    )
