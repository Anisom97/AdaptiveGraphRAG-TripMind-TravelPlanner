"""
Module 6 — FastAPI Backend
Exposes REST endpoints for the Streamlit frontend.
POST /generate — full itinerary generation pipeline
GET  /health   — system status check
GET  /stats    — vector store + KG stats
POST /ingest   — trigger ingestion pipeline
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

from m4_graphrag_fusion import UserPreferences
from m5_llm_composer import compose_itinerary
from m3_vector_store import get_collection_stats, build_vector_store


app = FastAPI(
    title="TripMind API",
    description="Hybrid KG + RAG itinerary planner — MTech thesis project",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────────

class ItineraryRequest(BaseModel):
    vibes:             list[str]      = Field(default_factory=list)
    budget_tier:       str            = "mid"
    duration_days:     int            = Field(default=7, ge=3, le=30)
    avoid_crowds:      bool           = False
    special_interests: list[str]      = Field(default_factory=list)
    regions:           list[str]      = Field(default_factory=list)
    travel_month:      Optional[str]  = None
    free_text:         str            = ""


class HealthResponse(BaseModel):
    status:          str
    vector_store:    str
    neo4j:           str
    ollama:          str
    total_documents: int


# ── Endpoints ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "TripMind API is running", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
def health_check():
    import requests as req

    # Vector store
    try:
        stats = get_collection_stats()
        vs_status = "ok"
        total_docs = stats["total_documents"]
    except Exception:
        vs_status = "unavailable"
        total_docs = 0

    # Neo4j
    try:
        from neo4j import GraphDatabase
        from m2_kg_builder import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as s:
            s.run("RETURN 1")
        driver.close()
        neo4j_status = "ok"
    except Exception:
        neo4j_status = "unavailable (run: neo4j start)"

    # Ollama
    try:
        r = req.get("http://localhost:11434/api/tags", timeout=3)
        ollama_status = "ok" if r.status_code == 200 else "unavailable"
    except Exception:
        ollama_status = "unavailable (run: ollama serve)"

    return HealthResponse(
        status="ok",
        vector_store=vs_status,
        neo4j=neo4j_status,
        ollama=ollama_status,
        total_documents=total_docs,
    )


@app.get("/stats")
def stats():
    try:
        vs = get_collection_stats()
    except Exception as e:
        vs = {"error": str(e)}
    return {"vector_store": vs}


@app.post("/generate")
def generate_itinerary(req: ItineraryRequest):
    """
    Main endpoint: runs full GraphRAG Fusion pipeline and returns itinerary JSON.
    """
    print("Itenary request")
    print(req)
    prefs = UserPreferences(
        vibes=req.vibes,
        budget_tier=req.budget_tier,
        duration_days=req.duration_days,
        avoid_crowds=req.avoid_crowds,
        special_interests=req.special_interests,
        regions=req.regions,
        travel_month=req.travel_month,
        free_text=req.free_text,
    )

    try:
        itinerary = compose_itinerary(prefs)
        return itinerary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
def trigger_ingestion(data_dir: str = "data/raw"):
    """Re-run the full ingestion pipeline (PDF → ChromaDB + Neo4j)."""
    try:
        build_vector_store(data_dir=data_dir)
        from m2_kg_builder import build_knowledge_graph
        build_knowledge_graph()
        return {"status": "ingestion complete"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Run ───────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("m6_api:app", host="0.0.0.0", port=8000, reload=True)
