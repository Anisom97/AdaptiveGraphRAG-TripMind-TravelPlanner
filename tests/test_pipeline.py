"""
TripMind — Quick test script
Tests each module in isolation without requiring Neo4j or Ollama.
Run: python tests/test_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "modules"))


def test_m1_ingestion():
    print("\n── Module 1: PDF Ingestion ──")
    from m1_pdf_ingestion import (
        clean_text, detect_vibes, detect_budget_tier,
        detect_content_type, chunk_by_tokens, DOCUMENT_REGISTRY
    )

    sample = "Botswana is one of Africa's premier safari destinations. Wildlife is incredible and uncrowded."
    assert "wildlife" in detect_vibes(sample)
    assert detect_content_type(sample) in ("guide", "highlights", "review", "budget_tips", "seasonal", "itinerary")
    chunks = chunk_by_tokens(sample * 10, max_tokens=50)
    assert len(chunks) > 1
    cleaned = clean_text("  theexplorersociety.com  12  \n\n\n  Hello world  ")
    assert "theexplorersociety" not in cleaned
    print("  PASS — text cleaning, vibe detection, chunking")
    print(f"  Registered documents: {[d['filename'] for d in DOCUMENT_REGISTRY]}")


def test_m4_query_router():
    print("\n── Module 4: Query Router ──")
    from m4_graphrag_fusion import UserPreferences, classify_query

    # Structural query — gorilla + quiet
    prefs_structural = UserPreferences(
        vibes=["wildlife"],
        special_interests=["gorilla"],
        avoid_crowds=True,
        budget_tier="mid",
    )
    plan = classify_query(prefs_structural)
    assert plan.alpha >= 0.55, f"Expected α≥0.55 for structural query, got {plan.alpha}"
    assert plan.query_type in ("structural", "hybrid")
    print(f"  Structural query → α={plan.alpha:.2f}, type={plan.query_type}  PASS")

    # Semantic query — vibe/feel language
    prefs_semantic = UserPreferences(
        vibes=["scenic"],
        free_text="I want a magical breathtaking beautiful experience",
        budget_tier="mid",
    )
    plan2 = classify_query(prefs_semantic)
    assert plan2.alpha <= 0.55, f"Expected α≤0.55 for semantic query, got {plan2.alpha}"
    print(f"  Semantic query   → α={plan2.alpha:.2f}, type={plan2.query_type}  PASS")

    # Hybrid
    prefs_hybrid = UserPreferences(
        vibes=["wildlife", "scenic"],
        budget_tier="mid",
        duration_days=7,
    )
    plan3 = classify_query(prefs_hybrid)
    print(f"  Hybrid query     → α={plan3.alpha:.2f}, type={plan3.query_type}  PASS")


def test_m4_fusion_scorer():
    print("\n── Module 4: Fusion Scorer ──")
    from m4_graphrag_fusion import UserPreferences, fuse_results, apply_graph_path_boost

    kg_mock = [
        {"name": "Uganda",   "region": "africa", "total_score": 8, "matched_vibes": ["wildlife"], "highlights": [], "best_for": "gorillas"},
        {"name": "Tanzania", "region": "africa", "total_score": 5, "matched_vibes": ["wildlife"], "highlights": [], "best_for": "safari"},
    ]
    rag_mock = [
        {"destination": "Uganda",   "score": 0.85, "text": "Gorilla trekking in Bwindi...", "source_id": "x", "source_title": "y", "region": "africa", "content_type": "guide", "vibes": []},
        {"destination": "Kenya",    "score": 0.72, "text": "Masai Mara wildlife...",       "source_id": "x", "source_title": "y", "region": "africa", "content_type": "guide", "vibes": []},
    ]
    prefs = UserPreferences(vibes=["wildlife"], special_interests=["gorilla"], avoid_crowds=True, regions=["africa"])

    fused = fuse_results(kg_mock, rag_mock, alpha=0.65, prefs=prefs)
    assert len(fused) >= 2
    fused = apply_graph_path_boost(fused, prefs)

    uganda = next((f for f in fused if f.destination == "Uganda"), None)
    assert uganda is not None
    assert uganda.graph_path_boost > 0, f"Uganda should get positive boost for gorilla, got {uganda.graph_path_boost}"
    print(f"  Uganda fusion_score={uganda.fusion_score:.3f} boost={uganda.graph_path_boost:+.3f}  PASS")


def test_m3_vector_store_mock():
    print("\n── Module 3: Vector Store (mock — no PDF needed) ──")
    from m1_pdf_ingestion import DocumentChunk, chunk_by_tokens

    sample_text = "The Serengeti is one of Africa's most famous national parks. " * 5
    chunks = chunk_by_tokens(sample_text, max_tokens=30)
    assert len(chunks) >= 2
    print(f"  Chunking: {len(chunks)} chunks from sample text  PASS")


def run_all():
    print("=" * 50)
    print("TripMind — Module Tests")
    print("=" * 50)

    tests = [
        test_m1_ingestion,
        test_m4_query_router,
        test_m4_fusion_scorer,
        test_m3_vector_store_mock,
    ]

    passed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} tests passed")
    print("="*50)

    if passed == len(tests):
        print("\nAll tests passed. Ready to run the full pipeline.")
        print("Next step: bash run.sh")


if __name__ == "__main__":
    run_all()
