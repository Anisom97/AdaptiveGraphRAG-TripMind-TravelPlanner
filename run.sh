#!/usr/bin/env bash
# TripMind — one-shot setup + run script
# Usage: bash run.sh

set -e

echo ""
echo "╔══════════════════════════════════════╗"
echo "║   TripMind — AI Itinerary Planner    ║"
echo "║   MTech AI & ML Thesis Project       ║"
echo "╚══════════════════════════════════════╝"
echo ""

# ── 1. Install Python dependencies ──────────────────────────────────────────────
echo "[1/5] Installing Python dependencies..."
pip install -r requirements.txt --quiet
python -m spacy download en_core_web_sm --quiet 2>/dev/null || true

# ── 2. Copy PDFs to data/raw ─────────────────────────────────────────────────────
echo "[2/5] Checking for PDF source files..."
mkdir -p data/raw

PDF_AFRICA="travel_guide_africa.pdf"
PDF_EUROPE="travel_guide_europe.pdf"

for pdf in "$PDF_AFRICA" "$PDF_EUROPE"; do
    if [ ! -f "data/raw/$pdf" ]; then
        if [ -f "$pdf" ]; then
            cp "$pdf" "data/raw/$pdf"
            echo "  Copied $pdf to data/raw/"
        else
            echo "  WARNING: $pdf not found. Place it in data/raw/ and re-run."
        fi
    else
        echo "  Found data/raw/$pdf"
    fi
done

# ── 3. Ingest PDFs → ChromaDB ─────────────────────────────────────────────────────
echo "[3/5] Running PDF ingestion + embedding..."
cd modules
python -c "
from m3_vector_store import build_vector_store, get_collection_stats
build_vector_store(data_dir='../data/raw')
stats = get_collection_stats()
print(f'  ChromaDB: {stats[\"total_documents\"]} documents stored')
"

# ── 4. Build Knowledge Graph ───────────────────────────────────────────────────────
echo "[4/5] Building Neo4j knowledge graph..."
echo "  NOTE: Neo4j must be running (neo4j start) for KG features."
echo "  If Neo4j is not running, the system will fallback to RAG-only mode."
python -c "
try:
    from m2_kg_builder import build_knowledge_graph
    build_knowledge_graph()
    print('  Neo4j knowledge graph built successfully')
except Exception as e:
    print(f'  Neo4j not available ({e}) — RAG-only mode will be used')
" 2>/dev/null || true

cd ..

# ── 5. Start servers ───────────────────────────────────────────────────────────────
echo "[5/5] Starting TripMind servers..."
echo ""
echo "  FastAPI backend → http://localhost:8000"
echo "  API docs        → http://localhost:8000/docs"
echo "  Streamlit UI    → http://localhost:8501"
echo ""
echo "  Press Ctrl+C to stop all servers."
echo ""

# Start FastAPI in background
cd modules
python m6_api.py &
API_PID=$!
cd ..

# Give API time to start
sleep 3

# Start Streamlit (foreground)
streamlit run frontend/app_frontend.py --server.port 8501 --server.headless true

# Cleanup on exit
kill $API_PID 2>/dev/null || true
