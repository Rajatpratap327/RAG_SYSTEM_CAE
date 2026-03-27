"""
Central configuration for the CAE RAG system.
All environment variables and defaults are managed here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Ollama ──────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# LLM for answer generation (text + reasoning)
LLM_MODEL: str = os.getenv("LLM_MODEL", "mistral-nemo")

# Vision Language Model for image summarisation
VLM_MODEL: str = os.getenv("VLM_MODEL", "llava")

# Embedding model (nomic-embed-text produces 768-dim embeddings locally)
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")

# ── ChromaDB ────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "cae_documents")

# ── Ingestion ────────────────────────────────────────────────────────────────
# Directory where extracted images are temporarily stored during ingestion
IMAGE_CACHE_DIR: str = os.getenv("IMAGE_CACHE_DIR", "./image_cache")
Path(IMAGE_CACHE_DIR).mkdir(parents=True, exist_ok=True)

# Maximum image dimension (px) before downscaling for VLM
MAX_IMAGE_DIM: int = int(os.getenv("MAX_IMAGE_DIM", "1024"))

# ── Retrieval ────────────────────────────────────────────────────────────────
# Number of chunks to retrieve per query
TOP_K: int = int(os.getenv("TOP_K", "6"))

# ── RAG Prompt ───────────────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT: str = """You are an expert CAE (Computer-Aided Engineering) analyst \
with deep knowledge of FEA (Finite Element Analysis), structural durability, bolted joint \
analysis per VDI 2230, and automotive engineering standards.

Answer the user's question using ONLY the context provided below. \
The context may contain text excerpts, table summaries, and descriptions of engineering \
diagrams or FEA contour plots.

Rules:
- Base your answer strictly on the provided context. Do not hallucinate.
- If the context contains a table relevant to the question, reference specific values.
- If the context contains an image description of an FEA plot or diagram, describe what it shows.
- Always cite your sources using the format [source: <filename>, page <page>, type: <chunk_type>].
- If the context does not contain enough information to answer, say so explicitly.

Context:
{context}
"""
