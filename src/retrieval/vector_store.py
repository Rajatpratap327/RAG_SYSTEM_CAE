"""
Vector store layer: ChromaDB with persistent storage.

ChromaDB is selected over FAISS because:
1. Native metadata filtering — we can retrieve chunks by type
   (text/table/image) without post-processing.
2. Persistent on-disk storage without serialisation boilerplate.
3. Collection-level management enables multi-document indexing with
   per-document deletion support.
"""

from typing import Any, Optional

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import CHROMA_COLLECTION, CHROMA_PERSIST_DIR
from src.ingestion.embedder import get_embedding_function


# Module-level singleton — one vector store instance per process
_vector_store: Optional[Chroma] = None


def get_vector_store() -> Chroma:
    """Return (and lazily initialise) the global ChromaDB vector store."""
    global _vector_store
    if _vector_store is None:
        _vector_store = Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=get_embedding_function(),
            persist_directory=CHROMA_PERSIST_DIR,
            collection_metadata={"hnsw:space": "cosine"},
        )
    return _vector_store


def add_chunks(chunks: list[dict[str, Any]]) -> int:
    """
    Embed and add parsed chunks to the vector store.

    Args:
        chunks: List of chunk dicts from the parser, each containing
                'text', 'metadata', and 'chunk_id'.

    Returns:
        Number of chunks successfully added.
    """
    vs = get_vector_store()
    documents = [
        Document(page_content=c["text"], metadata=c["metadata"])
        for c in chunks
    ]
    ids = [c["chunk_id"] for c in chunks]

    # ChromaDB will skip duplicates if the same ID already exists
    vs.add_documents(documents=documents, ids=ids)
    return len(documents)


def query_store(
    query: str,
    top_k: int = 6,
    chunk_type_filter: Optional[str] = None,
) -> list[Document]:
    """
    Retrieve the top-k most relevant chunks for a query.

    Args:
        query: Natural language question.
        top_k: Number of chunks to retrieve.
        chunk_type_filter: Optional filter — 'text', 'table', or 'image'.
                           When None, all chunk types are searched.

    Returns:
        List of LangChain Document objects with metadata.
    """
    vs = get_vector_store()
    search_kwargs: dict[str, Any] = {"k": top_k}

    if chunk_type_filter:
        search_kwargs["filter"] = {"chunk_type": {"$eq": chunk_type_filter}}

    retriever = vs.as_retriever(search_kwargs=search_kwargs)
    return retriever.invoke(query)


def get_store_stats() -> dict[str, Any]:
    """
    Return summary statistics about the current vector store state.

    Returns:
        Dict with document count, unique sources, and chunk type breakdown.
    """
    vs = get_vector_store()
    collection = vs._collection
    total = collection.count()

    # Fetch all metadata to compute breakdowns
    if total == 0:
        return {
            "total_chunks": 0,
            "unique_documents": 0,
            "chunk_type_breakdown": {"text": 0, "table": 0, "image": 0},
            "indexed_files": [],
        }

    results = collection.get(include=["metadatas"])
    metadatas = results.get("metadatas", [])

    sources: set[str] = set()
    breakdown = {"text": 0, "table": 0, "image": 0}

    for meta in metadatas:
        if meta:
            sources.add(meta.get("source", "unknown"))
            ctype = meta.get("chunk_type", "text")
            breakdown[ctype] = breakdown.get(ctype, 0) + 1

    return {
        "total_chunks": total,
        "unique_documents": len(sources),
        "chunk_type_breakdown": breakdown,
        "indexed_files": sorted(sources),
    }


def delete_document(filename: str) -> int:
    """
    Remove all chunks belonging to a specific source document.

    Args:
        filename: The PDF filename as stored in chunk metadata.

    Returns:
        Number of chunks deleted.
    """
    vs = get_vector_store()
    collection = vs._collection

    # Find IDs for this source
    results = collection.get(
        where={"source": {"$eq": filename}},
        include=["metadatas"],
    )
    ids_to_delete = results.get("ids", [])

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)

    return len(ids_to_delete)
