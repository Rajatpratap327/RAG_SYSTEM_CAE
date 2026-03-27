"""
Pydantic models for all FastAPI request and response schemas.
Explicit schemas improve Swagger auto-docs and enforce type safety.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Health endpoint ──────────────────────────────────────────────────────────

class ChunkTypeBreakdown(BaseModel):
    text: int = Field(..., description="Number of text chunks indexed")
    table: int = Field(..., description="Number of table chunks indexed")
    image: int = Field(..., description="Number of image-summary chunks indexed")


class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' if all models are reachable")
    llm_model: str = Field(..., description="Active LLM model name")
    vlm_model: str = Field(..., description="Active VLM model name")
    embed_model: str = Field(..., description="Active embedding model name")
    total_chunks: int = Field(..., description="Total chunks in the vector index")
    unique_documents: int = Field(..., description="Number of ingested documents")
    chunk_type_breakdown: ChunkTypeBreakdown
    indexed_files: list[str] = Field(..., description="List of ingested filenames")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")


# ── Ingest endpoint ──────────────────────────────────────────────────────────

class IngestionResponse(BaseModel):
    message: str
    filename: str
    total_chunks: int = Field(..., description="Total chunks added to the index")
    text_chunks: int
    table_chunks: int
    image_chunks: int
    processing_time_seconds: float


# ── Query endpoint ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        description="Natural language question about the ingested documents",
        examples=["What is the maximum von Mises stress reported in the FEA analysis?"],
    )
    top_k: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Number of chunks to retrieve (1–20)",
    )
    chunk_type_filter: Optional[str] = Field(
        default=None,
        description="Optional modality filter: 'text', 'table', or 'image'",
    )


class SourceReference(BaseModel):
    source: str = Field(..., description="Source filename")
    page: int = Field(..., description="Page number in the source PDF")
    chunk_type: str = Field(..., description="Chunk modality: text, table, or image")
    caption: str = Field(default="", description="Caption if available")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceReference]
    retrieved_chunks: int


# ── Documents endpoint ───────────────────────────────────────────────────────

class DocumentListResponse(BaseModel):
    indexed_files: list[str]
    total_documents: int


# ── Delete endpoint ──────────────────────────────────────────────────────────

class DeleteResponse(BaseModel):
    message: str
    filename: str
    chunks_deleted: int


# ── Error response ───────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    detail: str
