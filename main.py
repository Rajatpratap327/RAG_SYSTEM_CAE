"""
CAE Document Intelligence RAG System
Entry point for the FastAPI application.
"""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.retrieval.vector_store import get_vector_store

START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, clean up on shutdown."""
    # Initialise vector store on startup
    vs = get_vector_store()
    app.state.vector_store = vs
    app.state.start_time = START_TIME
    print("✅ Vector store initialised.")
    yield
    print("🛑 Shutting down CAE RAG server.")


app = FastAPI(
    title="CAE Document Intelligence RAG API",
    description=(
        "Multimodal Retrieval-Augmented Generation system for CAE/FEA engineering documents. "
        "Supports ingestion of PDFs containing text, tables, and engineering diagrams, "
        "with query capabilities grounded in retrieved context."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
