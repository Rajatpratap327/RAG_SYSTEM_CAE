# CAE Document Intelligence — Multimodal RAG System

> A fully on-premise Retrieval-Augmented Generation system for querying CAE/FEA engineering documents containing text, tables, and engineering diagrams.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Architecture Overview](#2-architecture-overview)
3. [Technology Choices](#3-technology-choices)
4. [Setup Instructions](#4-setup-instructions)
5. [API Documentation](#5-api-documentation)
6. [Screenshots](#6-screenshots)
7. [Limitations & Future Work](#7-limitations--future-work)

---

## 1. Problem Statement

### Domain

Automotive Computer-Aided Engineering (CAE) — specifically structural durability analysis, Finite Element Analysis (FEA), and bolted joint validation per VDI 2230.

### The Problem

In automotive structural engineering, CAE analysts routinely produce and consume large volumes of technical PDF documents: FEA simulation reports, durability test summaries, design validation records, and standards documents such as VDI 2230 (systematic calculation of high duty bolted joints). These documents are intrinsically multimodal — they combine dense technical prose, specification tables (material properties, torque values, safety factors, load cases), and engineering images (von Mises stress contour plots, load-displacement curves, bolt geometry cross-sections, fatigue life charts).

Currently, querying this document corpus is a manual and time-consuming process. An analyst asking "What is the minimum safety factor against yielding reported across all load cases in the steering knuckle FEA report?" must manually open each PDF, scan tables, and cross-reference values — a task that can take 30–60 minutes per query across a typical project document set of 20–50 reports. The problem is compounded by the fact that critical information is often split across modalities: a table may report raw stress values while the associated contour plot communicates spatial distribution and failure location — and both are needed to answer a complete engineering query.

Traditional keyword search fails because engineering terminology is highly specialised (e.g., "Rp0.2", "Fv", "αA", "interface normal force"), abbreviations are non-standard across organisations, and the most critical information is locked inside tables and images that keyword indexers cannot parse.

### Why This Problem Is Unique

Unlike a generic document Q&A system, CAE document retrieval presents specific challenges:

- **Table density:** FEA reports routinely contain tables with 10–20 columns of simulation outputs (nodal forces, reaction moments, safety factors per VDI 2230 criteria). A system that only indexes text paragraphs misses the majority of quantitative results.
- **Image semantics:** Contour plots of von Mises stress or principal strain are not decorative — they carry primary engineering findings. A system that discards images loses the spatial failure mode information that analysts depend on.
- **Cross-modal answers:** Many engineering queries require combining information from different modalities. "Does the bolt meet the VDI 2230 SF requirement?" requires reading the safety factor value from a results table *and* confirming the load application from an FBD diagram.
- **Specialised vocabulary:** Standard semantic search models are not trained on FEA terminology. Domain-tuned embeddings or retrieval with strong prompt engineering are required for meaningful results.

### Why RAG Is the Right Approach

Fine-tuning a language model on CAE documents is impractical: the document corpus changes with every project cycle, retraining is expensive, and the model would need to memorise specific numerical values that must be grounded in source documents for regulatory traceability (IATF 16949, internal DVP requirements). Keyword search cannot handle semantic variation in how analysts phrase queries. Manual search does not scale.

RAG directly addresses these constraints: it retrieves factually grounded context from the actual documents at query time, requires no retraining as the corpus evolves, and returns source references that satisfy traceability requirements. The multimodal extension — processing tables as structured text and images as VLM-generated summaries before embedding — ensures that no modality is silently ignored.

### Expected Outcomes

A successful system enables analysts to:

- Query a corpus of FEA reports and receive grounded answers with page-level citations in under 30 seconds.
- Ask table-specific questions such as "What torque preload was applied to M10 bolts in the load case 3 analysis?" and receive the exact value from the indexed table.
- Ask image-specific questions such as "What thread geometry parameters are shown in the diagram?" and receive a description grounded in the VLM-summarised engineering figure.
- Perform cross-document synthesis: "Across all ingested durability reports, which component has the lowest fatigue safety factor?"

---

## 2. Architecture Overview

### Ingestion Pipeline

```mermaid
flowchart TD
    A[PDF Upload\n/ingest] --> B[Docling Parser]
    B --> C1[Text Blocks]
    B --> C2[Tables → Markdown]
    B --> C3[Images → PNG]
    C3 --> D[minicpm-v VLM\nOllama]
    D --> E[Image Summary Text]
    C1 --> F[nomic-embed-text\nOllama]
    C2 --> F
    E --> F
    F --> G[ChromaDB\nVector Store\nPersisted on Disk]
```

### Query Pipeline

```mermaid
flowchart TD
    A[User Question\nPOST /query] --> B[nomic-embed-text\nEmbed Query]
    B --> C[ChromaDB\nSimilarity Search\ntop-k chunks]
    C --> D[Context Formatter\ntext + table + image chunks]
    D --> E[Mistral-Nemo\nOllama\nCAE System Prompt]
    E --> F[Answer + Source References\nJSON Response]
```

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Server                        │
│   /health  /ingest  /query  /documents  DELETE          │
└────────────────────────┬────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
┌────────▼───────┐ ┌─────▼──────┐ ┌─────▼──────────┐
│   Docling      │ │  ChromaDB  │ │  Ollama Models  │
│   PDF Parser   │ │  (local    │ │                 │
│   text/table/  │ │  chroma_db)│ │  mistral-nemo   │
│   image chunks │ │            │ │  minicpm-v      │
└────────────────┘ └────────────┘ │  nomic-embed    │
                                   └─────────────────┘
```

---

## 3. Technology Choices

| Component | Choice | Justification |
|---|---|---|
| **Document Parser** | Docling 2.x | Native support for table extraction to DataFrame (→ Markdown), picture extraction as PIL images, and OCR via EasyOCR — all in one pipeline without stitching multiple tools. |
| **Embedding Model** | `nomic-embed-text` (Ollama) | 768-dim embeddings, optimised for long-form retrieval, fully local with no API key required. Outperforms `all-MiniLM` on domain-specific retrieval benchmarks. Chunk size capped at 800 characters to respect the model context window. |
| **Vector Store** | ChromaDB | Native `where` clause metadata filtering enables retrieval by chunk type (`text`/`table`/`image`) without post-processing. Persistent on-disk storage without serialisation boilerplate. FAISS would require manual metadata management and lacks native filtering. |
| **LLM** | Mistral-Nemo (Ollama) | 12B parameter model with 128k context window. Strong instruction following for grounded QA. Fully on-premise — no data leaves the machine. Low temperature (0.1) used for deterministic engineering answers. |
| **Vision Model** | minicpm-v (Ollama) | Chosen over LLaVA for significantly lower RAM requirements (~4.7GB vs ~8GB) making it viable on 8GB RAM systems. Produces detailed engineering image descriptions including thread geometry, bolt assembly diagrams, and mechanical schematics. Offline CMD inference used when server RAM is constrained. |
| **Framework** | LangChain + FastAPI | LangChain provides the retriever abstraction over ChromaDB and the message interface for Ollama. FastAPI provides Pydantic schema validation, automatic Swagger docs, and async file handling. |
| **Text Splitting** | RecursiveCharacterTextSplitter | Added to handle long text and table chunks that exceed nomic-embed-text context window. Chunk size 800 chars with 100 char overlap preserves context across splits. |

---

## 4. Setup Instructions

### Prerequisites

- Python 3.11+ (Python 3.14 may cause pandas compilation issues — 3.11 recommended)
- [Ollama](https://ollama.com) installed and running
- 8 GB RAM minimum (16 GB recommended for simultaneous VLM + LLM inference)
- Windows Developer Mode enabled (required for Docling model symlinks on Windows)

### Step 1 — Enable Windows Developer Mode (Windows only)

Settings → System → For Developers → toggle **Developer Mode** ON → restart machine.

This is required for Docling to create model symlinks in the HuggingFace cache.

### Step 2 — Pull required Ollama models

```bash
ollama pull mistral-nemo
ollama pull minicpm-v
ollama pull nomic-embed-text
```

Verify all three are available:

```bash
ollama list
```

### Step 3 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/cae-rag-system.git
cd cae-rag-system
```

### Step 4 — Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### Step 5 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 6 — Configure environment

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # Linux/macOS
```

Default `.env` values work out of the box. Edit only if your Ollama URL or model names differ.

### Step 7 — Start the server

```bash
python main.py
```

The server starts at `http://localhost:8000`.

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### Step 8 — Ingest the sample document

Via Swagger UI at `http://localhost:8000/docs` → POST /ingest → Try it out → upload PDF.

Or via curl:
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@sample_documents/IJSER151593.pdf"
```

> **Note on 8GB RAM systems:** Close all other applications before ingesting. minicpm-v requires ~4.7GB RAM. For best image description quality, run minicpm-v in CMD first to warm it up before starting the server:
> ```bash
> ollama run minicpm-v "ready"
> ```
> Then start the server in a second terminal immediately.

### Step 9 — Run a test query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What thread geometry parameters are defined in the bolt analysis?"}'
```

---

## 5. API Documentation

### GET /health

Returns system status, model configuration, and index statistics.

**Response example:**
```json
{
  "status": "ok",
  "llm_model": "mistral-nemo",
  "vlm_model": "minicpm-v",
  "embed_model": "nomic-embed-text",
  "total_chunks": 100,
  "unique_documents": 1,
  "chunk_type_breakdown": {
    "text": 95,
    "table": 1,
    "image": 4
  },
  "indexed_files": ["IJSER151593.pdf"],
  "uptime_seconds": 124.3
}
```

---

### POST /ingest

Upload a PDF file for parsing and indexing.

**Request:** `multipart/form-data` with field `file` (PDF only).

**Response example:**
```json
{
  "message": "Successfully ingested 'IJSER151593.pdf'",
  "filename": "IJSER151593.pdf",
  "total_chunks": 100,
  "text_chunks": 95,
  "table_chunks": 1,
  "image_chunks": 4,
  "processing_time_seconds": 1109.17
}
```

**Error responses:**
- `400` — Non-PDF file uploaded
- `422` — PDF parsing failed (corrupted or password-protected file)

---

### POST /query

Query the indexed documents with a natural language question.

**Request body:**
```json
{
  "question": "What thread geometry parameters are defined for bolt analysis?",
  "top_k": 6,
  "chunk_type_filter": null
}
```

`chunk_type_filter` is optional. When provided, must be `"text"`, `"table"`, or `"image"`.

**Response example:**
```json
{
  "question": "What thread geometry parameters are defined for bolt analysis?",
  "answer": "The document defines the following thread geometry parameters for bolt analysis per VDI 2230: pitch diameter d2, minor diameter d1, major diameter d, root radius, flank angle, and thread height H. These parameters are used to compute shear stress area, preload capacity, and thread stripping safety factors. [source: IJSER151593.pdf, page 1, type: image]",
  "sources": [
    {
      "source": "IJSER151593.pdf",
      "page": 1,
      "chunk_type": "image",
      "caption": ""
    }
  ],
  "retrieved_chunks": 6
}
```

---

### GET /documents

List all currently indexed documents.

**Response example:**
```json
{
  "indexed_files": ["IJSER151593.pdf"],
  "total_documents": 1
}
```

---

### DELETE /documents/{filename}

Remove all chunks for a specific document from the index.

**Example:** `DELETE /documents/IJSER151593.pdf`

**Response example:**
```json
{
  "message": "Successfully removed 'IJSER151593.pdf' from the index.",
  "filename": "IJSER151593.pdf",
  "chunks_deleted": 100
}
```

---

### GET /docs

FastAPI auto-generated Swagger/OpenAPI UI. Access at `http://localhost:8000/docs`.

---

## 6. Screenshots

> All screenshots are located in the `screenshots/` folder.

| # | File | Description |
|---|---|---|
| 1 | `screenshots/01_swagger_ui.png` | `/docs` Swagger UI showing all 5 endpoints |
| 2 | `screenshots/02_ingest_response.png` | POST `/ingest` response — 100 chunks (95 text, 1 table, 4 image) |
| 3 | `screenshots/03_text_query.png` | Text chunk query result with source references |
| 4 | `screenshots/04_table_query.png` | Table chunk query with `chunk_type_filter: "table"` |
| 5 | `screenshots/05_image_query.png` | Image chunk query with `chunk_type_filter: "image"` |
| 6 | `screenshots/06_health_endpoint.png` | `/health` response showing 100 indexed chunks |

---

## 7. Limitations & Future Work

### Current Limitations

- **VLM RAM constraint:** minicpm-v requires ~4.7GB RAM. On 8GB systems, the Python server and minicpm-v cannot run simultaneously. Workaround implemented: VLM is run offline via CMD before server startup to warm up RAM, or manual descriptions are used as a fallback for known documents. A GPU-equipped machine eliminates this constraint entirely.
- **Image description quality:** When VLM times out, a structured placeholder description is stored for image chunks. While the placeholder preserves retrieval capability, it reduces image query answer quality compared to real VLM inference.
- **OCR quality:** Docling's EasyOCR is effective for printed text but degrades on low-resolution scanned PDFs or documents with complex multi-column layouts common in older SAE/VDI standards.
- **Context window constraints:** nomic-embed-text context is limited — chunks are capped at 800 characters. Very long tables are split across multiple chunks which may reduce table retrieval precision.
- **No re-ranking:** Retrieved chunks are ranked purely by cosine similarity. A cross-encoder re-ranker (e.g., `ms-marco-MiniLM`) would improve precision, particularly for table retrieval.
- **Single-turn QA only:** The `/query` endpoint does not maintain conversational context between calls. Each query is stateless.
- **No authentication:** The API has no access control. In a production deployment, OAuth2 or API key middleware should be added.
- **Windows symlink requirement:** Docling requires Windows Developer Mode to create HuggingFace model symlinks. This is a one-time setup step but may be a barrier in enterprise environments.

### Future Work

- **GPU acceleration:** Containerise with NVIDIA CUDA base image to run minicpm-v and mistral-nemo simultaneously without RAM constraints.
- **Re-ranking layer:** Add a cross-encoder re-ranker between retrieval and generation to improve answer quality on ambiguous queries.
- **Conversational memory:** Add a `/chat` endpoint that maintains session-level conversation history using LangChain's `ConversationBufferMemory`.
- **Structured output parsing:** For table queries, return structured JSON values rather than prose — enabling downstream integration with engineering calculation tools.
- **Evaluation harness:** Implement RAGAS-based automatic evaluation (faithfulness, context recall, answer relevancy) against a manually curated golden QA set from real CAE reports.
- **Multi-vector retrieval:** Embed table summaries and raw table markdown separately to improve table retrieval precision.
- **Docker deployment:** Package the entire stack (FastAPI + Ollama + ChromaDB) in Docker Compose for one-command deployment on any machine.

---

*Built for BITS Pilani WILP — Multimodal RAG Bootcamp Assignment*
*Domain: Automotive CAE / Structural Durability Engineering*