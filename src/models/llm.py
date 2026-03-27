"""
LLM wrapper for answer generation.
Uses Mistral-Nemo via Ollama for RAG answer synthesis.
"""

from langchain_ollama import ChatOllama

from src.config import LLM_MODEL, OLLAMA_BASE_URL


def get_llm(temperature: float = 0.1) -> ChatOllama:
    """
    Return a LangChain ChatOllama instance configured for Mistral-Nemo.

    Mistral-Nemo is chosen for its strong instruction-following and
    128k context window, suitable for multi-chunk CAE document answers.

    Args:
        temperature: Sampling temperature. Low values (0.1) favour
                     factual, deterministic answers — appropriate for
                     engineering document QA.
    """
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
    )
