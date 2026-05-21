"""
Module 3 — Vector Store Builder
Embeds document chunks using all-MiniLM-L6-v2 (CPU-friendly, 80MB).
Stores in ChromaDB local persistent store.
"""

import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from m1_pdf_ingestion import ingest_all, DocumentChunk


CHROMA_PATH   = os.getenv("CHROMA_PATH", "data/chroma_db")
COLLECTION_NAME = "travel_content"
EMBED_MODEL   = "all-MiniLM-L6-v2"

_model  = None
_client = None
_collection = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBED_MODEL} ...")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def embed_chunks(chunks: list[DocumentChunk], batch_size: int = 64):
    """Embed all chunks and upsert into ChromaDB."""
    model      = get_model()
    collection = get_collection()

    total = len(chunks)
    print(f"Embedding {total} chunks in batches of {batch_size}...")

    for i in range(0, total, batch_size):
        batch = chunks[i: i + batch_size]
        texts = [c.text for c in batch]
        ids   = [c.chunk_id for c in batch]
        metas = [
            {
                "source_id":    c.source_id,
                "source_title": c.source_title,
                "region":       c.region,
                "doc_type":     c.doc_type,
                "destination":  c.destination,
                "content_type": c.content_type,
                "vibes":        ",".join(c.vibes),
                "budget_tier":  c.budget_tier or "unknown",
                "has_best_for": str(c.has_best_for),
                "page_num":     str(c.page_num),
            }
            for c in batch
        ]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metas,
        )
        print(f"  Upserted batch {i // batch_size + 1} / {(total + batch_size - 1) // batch_size}")

    print(f"Vector store ready: {collection.count()} documents in ChromaDB")


def build_vector_store(data_dir: str = "data/raw"):
    """Full pipeline: ingest PDFs → embed → store."""
    chunks = ingest_all(data_dir=data_dir)
    if not chunks:
        print("No chunks to embed. Check PDF paths.")
        return
    embed_chunks(chunks)


def semantic_search(
    query: str,
    n_results: int = 10,
    region_filter: str = None,
    destination_filter: str = None,
    content_type_filter: str = None,
) -> list[dict]:
    """
    Embed query and retrieve top-k relevant chunks from ChromaDB.
    Optional metadata filters narrow the search space.
    """
    model      = get_model()
    collection = get_collection()

    query_embedding = model.encode([query]).tolist()

    where_clause = {}
    if region_filter:
        where_clause["region"] = region_filter
    if destination_filter:
        where_clause["destination"] = destination_filter
    if content_type_filter:
        where_clause["content_type"] = content_type_filter

    kwargs = dict(
        query_embeddings=query_embedding,
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    if where_clause:
        kwargs["where"] = where_clause

    results = collection.query(**kwargs)

    output = []
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        output.append({
            "text":         doc,
            "source_id":    meta["source_id"],
            "source_title": meta["source_title"],
            "destination":  meta["destination"],
            "region":       meta["region"],
            "content_type": meta["content_type"],
            "vibes":        meta["vibes"].split(",") if meta["vibes"] else [],
            "score":        round(1 - dist, 4),   # cosine similarity
        })

    return output


def get_collection_stats() -> dict:
    col = get_collection()
    return {"total_documents": col.count(), "collection_name": COLLECTION_NAME}


if __name__ == "__main__":
    build_vector_store(data_dir="data/raw")

    print("\nTest search: 'gorilla trekking quiet forest Uganda'")
    hits = semantic_search("gorilla trekking quiet forest Uganda", n_results=5)
    for h in hits:
        print(f"  [{h['score']:.3f}] {h['destination']} | {h['text'][:80]}...")
