"""
RAG chain: retrieves relevant chunks and generates grounded answers
using Mistral-Nemo with a domain-specific CAE prompt template.
"""

from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import RAG_SYSTEM_PROMPT, TOP_K
from src.models.llm import get_llm
from src.retrieval.vector_store import query_store


def _format_context(docs: list[Document]) -> str:
    """
    Format retrieved documents into a single context string for the LLM.

    Each chunk is labelled with its type and source for traceability.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        chunk_type = meta.get("chunk_type", "text")
        header = f"[{i}] Source: {source} | Page: {page} | Type: {chunk_type}"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _build_source_references(docs: list[Document]) -> list[dict[str, Any]]:
    """Extract structured source references from retrieved documents."""
    refs = []
    for doc in docs:
        meta = doc.metadata
        refs.append(
            {
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", 0),
                "chunk_type": meta.get("chunk_type", "text"),
                "caption": meta.get("caption", ""),
            }
        )
    return refs


def run_rag_query(
    question: str,
    top_k: int = TOP_K,
    chunk_type_filter: str | None = None,
) -> dict[str, Any]:
    """
    Execute the full RAG pipeline for a user question.

    Pipeline steps:
    1. Embed the question and retrieve top-k relevant chunks.
    2. Format chunks into a structured context string.
    3. Build a system + user message pair and invoke Mistral-Nemo.
    4. Return the answer with source references and retrieval metadata.

    Args:
        question: Natural language question from the user.
        top_k: Number of chunks to retrieve.
        chunk_type_filter: Optional modality filter ('text', 'table', 'image').

    Returns:
        Dict with 'answer', 'sources', and 'retrieved_chunks' fields.

    Raises:
        ValueError: If no chunks are found in the vector store.
    """
    # Step 1: Retrieve
    docs = query_store(question, top_k=top_k, chunk_type_filter=chunk_type_filter)

    if not docs:
        return {
            "answer": (
                "No relevant documents found in the index. "
                "Please ingest domain PDF documents before querying."
            ),
            "sources": [],
            "retrieved_chunks": 0,
        }

    # Step 2: Build context
    context = _format_context(docs)

    # Step 3: Generate answer
    llm = get_llm()
    system_msg = SystemMessage(content=RAG_SYSTEM_PROMPT.format(context=context))
    human_msg = HumanMessage(content=question)

    response = llm.invoke([system_msg, human_msg])
    answer = response.content.strip()

    # Step 4: Return structured result
    return {
        "answer": answer,
        "sources": _build_source_references(docs),
        "retrieved_chunks": len(docs),
    }
