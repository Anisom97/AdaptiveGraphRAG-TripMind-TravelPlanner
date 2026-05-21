# TripMind — AI Itinerary Planner
**MTech AI & ML Thesis Project — Hybrid KG + RAG Travel Itinerary System**

---

## Architecture

```
tripmind/
├── modules/
│   ├── m1_pdf_ingestion.py     ← PDF → clean chunks
│   ├── m2_kg_builder.py        ← Chunks → Neo4j knowledge graph
│   ├── m3_vector_store.py      ← Chunks → ChromaDB embeddings
│   ├── m4_graphrag_fusion.py   ← ★ Research contribution: Query Router + Fusion Scorer + Re-ranker
│   ├── m5_llm_composer.py      ← Retrieval → Ollama Mistral → itinerary JSON
│   └── m6_api.py               ← FastAPI REST backend
├── frontend/
│   └── app.py                  ← Streamlit UI
├── data/
│   ├── raw/                    ← Put your PDFs here
│   └── chroma_db/              ← Auto-created by ChromaDB
├── tests/
│   └── test_pipeline.py        ← Unit tests (no external services needed)
├── requirements.txt
├── run.sh                      ← One-shot setup + start
└── README.md
```

---

## Quick Start

### 1. Prerequisites

```bash
# Python 3.10+
python --version

# Ollama (for local LLM)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral     # ~4GB download, runs in ~5-6GB RAM

# Neo4j Desktop (free)
# Download from: https://neo4j.com/download/
# Create a local database, set password to: tripmind123
# Start the database before running TripMind
```

### 2. Add your PDFs

```bash
cp travel_guide_africa.pdf data/raw/
cp travel_guide_europe.pdf data/raw/
```

### 3. Run everything

```bash
chmod +x run.sh
bash run.sh
```

This will:
- Install all Python dependencies
- Extract and embed PDF content into ChromaDB
- Build the Neo4j knowledge graph
- Start FastAPI on `http://localhost:8000`
- Start Streamlit UI on `http://localhost:8501`

### 4. Run tests (no external services needed)

```bash
python tests/test_pipeline.py
```

---

## Manual startup (step by step)

```bash
# Terminal 1: Start Neo4j (via Neo4j Desktop or CLI)
neo4j start

# Terminal 2: Start Ollama
ollama serve

# Terminal 3: Ingest PDFs
cd modules
python m1_pdf_ingestion.py    # extract + chunk
python m2_kg_builder.py       # build Neo4j graph
python m3_vector_store.py     # embed → ChromaDB

# Terminal 4: Start API
cd modules
python m6_api.py

# Terminal 5: Start UI
streamlit run frontend/app.py
```

---

## Module descriptions

| Module | File | Role |
|--------|------|------|
| M1 | `m1_pdf_ingestion.py` | PyMuPDF text extraction, cleaning, section-aware chunking |
| M2 | `m2_kg_builder.py` | SpaCy NER + static seed → Neo4j destination/vibe/season/animal graph |
| M3 | `m3_vector_store.py` | all-MiniLM-L6-v2 embeddings → ChromaDB persistent store |
| M4 | `m4_graphrag_fusion.py` | **Research contribution**: Query Router + Weighted Fusion Scorer + Graph-Path Re-ranker |
| M5 | `m5_llm_composer.py` | LangChain prompt builder → Ollama Mistral-7B → structured itinerary JSON |
| M6 | `m6_api.py` | FastAPI: `/generate`, `/health`, `/stats`, `/ingest` |
| M7 | `frontend/app.py` | Streamlit: preference panel + fusion debug + day-by-day itinerary cards |

---

## Research contribution (Module 4)

The novel contribution is the **GraphRAG Fusion module**, which consists of:

### 1. Query Router
Classifies each query as structural, semantic, or hybrid by counting signal keywords.
Hard constraints (gorilla-only destinations, crowd avoidance, specific months) push α toward KG.

### 2. Fusion Scorer
```
fusion_score = α × KG_score_normalised + (1-α) × RAG_score_normalised
```
α is dynamically set per query (not a fixed hyperparameter).

### 3. Contextual Re-ranker
Uses graph-path distance from the destination node to requested interest nodes.
A destination that is 1-hop from "gorilla" (e.g., Uganda, Rwanda) gets a +0.15 boost.
Destinations in wrong regions get a -0.20 penalty.

### Adaptive fallback
If Neo4j returns 0 results (e.g., very niche query), α is set to 0.0 and the system falls back to pure RAG retrieval.

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `tripmind123` | Neo4j password |
| `CHROMA_PATH` | `data/chroma_db` | ChromaDB storage path |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API URL |
| `OLLAMA_MODEL` | `mistral` | Model name (use `llama3.2` for smaller RAM) |

---

## RAM budget (mid-range machine, 8–16 GB)

| Component | RAM |
|-----------|-----|
| Mistral-7B Q4 (Ollama) | ~5.5 GB |
| Neo4j local | ~500 MB |
| ChromaDB + MiniLM | ~400 MB |
| FastAPI | ~150 MB |
| Streamlit | ~200 MB |
| **Total** | **~6.8 GB** |

> If you have only 8 GB RAM, use `ollama pull llama3.2:3b` instead of mistral — set `OLLAMA_MODEL=llama3.2:3b`.

---

## Data sources

- `travel_guide_africa.pdf` — The Explorer Society: Ultimate Guide to a Safari in Africa (2023)
- `travel_guide_europe.pdf` — Wandering Lewis: Ultimate Quick Travel Guide European Edition

---

## Thesis citation note

When writing your thesis, cite the data sources and note that:
- The KG seed data was derived from both PDFs using SpaCy NER + manual curation
- The vector store contains chunks from both documents
- Evaluation baselines: RAG-only vs KG-only vs GraphRAG Fusion (M4)
