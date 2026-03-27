"""
Embedding layer: wraps Ollama's nomic-embed-text model as a
LangChain-compatible embedding interface for ChromaDB ingestion.
"""

from langchain_ollama import OllamaEmbeddings

from src.config import EMBED_MODEL, OLLAMA_BASE_URL


def get_embedding_function() -> OllamaEmbeddings:
    """
    Return a LangChain OllamaEmbeddings instance configured for
    the nomic-embed-text model running locally.

    nomic-embed-text produces 768-dimensional embeddings and is
    optimised for long-form document retrieval tasks.
    """
    return OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
