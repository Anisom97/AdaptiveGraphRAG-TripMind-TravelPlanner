"""
Module 4 — GraphRAG Fusion  ★ Research Contribution ★
Three-part novel hybrid retrieval:
  1. Query Router     — classifies structural vs semantic intent
  2. Fusion Scorer    — weighted merge α·KG + (1-α)·RAG
  3. Contextual Re-ranker — graph-path distance boosts RAG chunks

This is the core of the MTech thesis.
"""

import re
from dataclasses import dataclass, field
from m2_kg_builder import query_destinations
from m3_vector_store import semantic_search


# ── Data types ───────────────────────────────────────────────────────────────────

@dataclass
class UserPreferences:
    vibes: list[str]         = field(default_factory=list)
    budget_tier: str         = "mid"
    duration_days: int       = 7
    avoid_crowds: bool       = False
    special_interests: list  = field(default_factory=list)
    regions: list[str]       = field(default_factory=list)
    travel_month: str        = None
    free_text: str           = ""


@dataclass
class QueryPlan:
    query_type: str          # "structural", "semantic", "hybrid"
    alpha: float             # KG weight (0=all RAG, 1=all KG)
    kg_vibes: list[str]
    rag_query: str
    reasoning: str


@dataclass
class FusedResult:
    destination: str
    region: str
    kg_score: float
    rag_score: float
    fusion_score: float
    matched_vibes: list[str]
    highlights: list[str]
    rag_chunks: list[dict]
    best_for: str
    graph_path_boost: float


# ── 1. QUERY ROUTER ───────────────────────────────────────────────────────────────

# Structural keywords → strong KG signal (relationships, filters)
STRUCTURAL_SIGNALS = [
    "gorilla", "gorillas", "specific animal", "big five", "penguins", "meerkats",
    "whale", "chimpanzee", "only in", "where can i find",
    "budget", "cheap", "luxury", "expensive",
    "month", "season", "when", "best time",
    "quiet", "crowded", "no crowds", "isolation",
    "beginner", "experienced", "first time",
]

# Semantic keywords → strong RAG signal (vibe, experience, description)
SEMANTIC_SIGNALS = [
    "vibe", "feel", "atmosphere", "like", "similar", "recommend",
    "describe", "what is it like", "experience", "adventure",
    "romantic", "magical", "stunning", "beautiful", "breathtaking",
    "tell me about", "hidden gem", "off the beaten",
]


def classify_query(prefs: UserPreferences) -> QueryPlan:
    """
    Analyse user preferences to decide how to weight KG vs RAG retrieval.
    Returns a QueryPlan with alpha and constructed queries.
    """
    combined_text = " ".join(prefs.vibes + prefs.special_interests + [prefs.free_text]).lower()

    structural_hits = sum(1 for s in STRUCTURAL_SIGNALS if s in combined_text)
    semantic_hits   = sum(1 for s in SEMANTIC_SIGNALS   if s in combined_text)

    # Hard constraints always push toward KG
    hard_structural = any([
        prefs.avoid_crowds,
        prefs.travel_month is not None,
        len(prefs.special_interests) > 0,
        prefs.budget_tier in ("budget", "luxury"),
    ])

    if hard_structural:
        structural_hits += 2

    total = structural_hits + semantic_hits
    if total == 0:
        alpha = 0.5
        query_type = "hybrid"
    elif structural_hits > semantic_hits * 1.5:
        alpha = 0.70
        query_type = "structural"
    elif semantic_hits > structural_hits * 1.5:
        alpha = 0.30
        query_type = "semantic"
    else:
        alpha = 0.55
        query_type = "hybrid"

    # Build RAG query string from preferences
    rag_parts = []
    if prefs.vibes:
        rag_parts.append(" ".join(prefs.vibes))
    if prefs.special_interests:
        rag_parts.append(" ".join(prefs.special_interests))
    if prefs.free_text:
        rag_parts.append(prefs.free_text)
    if prefs.avoid_crowds:
        rag_parts.append("quiet uncrowded remote wilderness")
    if prefs.budget_tier == "budget":
        rag_parts.append("budget affordable cheap backpacker")
    elif prefs.budget_tier == "luxury":
        rag_parts.append("luxury premium private exclusive lodge")

    rag_query = " ".join(rag_parts) or "travel destination guide"

    reasoning = (
        f"Structural signals: {structural_hits} | Semantic signals: {semantic_hits} | "
        f"Hard constraints: {hard_structural} → α={alpha:.2f} ({query_type})"
    )

    return QueryPlan(
        query_type=query_type,
        alpha=alpha,
        kg_vibes=prefs.vibes + (["uncrowded"] if prefs.avoid_crowds else []),
        rag_query=rag_query,
        reasoning=reasoning,
    )


# ── 2. FUSION SCORER ──────────────────────────────────────────────────────────────

def normalize_scores(items: list, score_key: str, max_val: float = None) -> list:
    """Min-max normalise a list of dicts on score_key → [0, 1]."""
    scores = [i[score_key] for i in items]
    if not scores:
        return items
    mn, mx = min(scores), max(scores) if max_val is None else max_val
    if mx == mn:
        for i in items:
            i[f"{score_key}_norm"] = 1.0
    else:
        for i in items:
            i[f"{score_key}_norm"] = (i[score_key] - mn) / (mx - mn)
    return items


def fuse_results(
    kg_results: list[dict],
    rag_results: list[dict],
    alpha: float,
    prefs: UserPreferences,
) -> list[FusedResult]:
    """
    Merge KG and RAG results using weighted fusion score.
    fusion_score = α · kg_norm + (1-α) · rag_norm
    """
    # Normalise KG scores
    kg_results = normalize_scores(
        [{**r, "total_score": r.get("total_score", 0)} for r in kg_results],
        "total_score",
    )

    # Build destination → rag_chunks mapping from RAG hits
    dest_rag_map: dict[str, list] = {}
    for chunk in rag_results:
        d = chunk["destination"]
        dest_rag_map.setdefault(d, []).append(chunk)

    # Compute per-destination RAG score = max chunk similarity
    dest_rag_score: dict[str, float] = {}
    for d, chunks in dest_rag_map.items():
        dest_rag_score[d] = max(c["score"] for c in chunks)

    # Collect all destination names
    all_dests = set(r["name"] for r in kg_results) | set(dest_rag_score.keys())

    fused = []
    for dest in all_dests:
        kg_row  = next((r for r in kg_results if r["name"] == dest), None)
        kg_norm = kg_row["total_score_norm"] if kg_row else 0.0
        rag_norm = dest_rag_score.get(dest, 0.0)

        fusion_score = alpha * kg_norm + (1 - alpha) * rag_norm

        fused.append(FusedResult(
            destination=dest,
            region=kg_row["region"] if kg_row else "unknown",
            kg_score=round(kg_norm, 4),
            rag_score=round(rag_norm, 4),
            fusion_score=round(fusion_score, 4),
            matched_vibes=kg_row["matched_vibes"] if kg_row else [],
            highlights=kg_row["highlights"] if kg_row else [],
            rag_chunks=dest_rag_map.get(dest, [])[:4],
            best_for=kg_row["best_for"] if kg_row else "",
            graph_path_boost=0.0,
        ))

    fused.sort(key=lambda x: x.fusion_score, reverse=True)
    return fused


# ── 3. CONTEXTUAL RE-RANKER ───────────────────────────────────────────────────────

def apply_graph_path_boost(
    fused: list[FusedResult],
    prefs: UserPreferences,
) -> list[FusedResult]:
    """
    Graph-path distance re-ranking:
    Destinations that are 1-hop from ALL requested special interests get a boost.
    Destinations 2+ hops away get a penalty.
    This simulates graph-path distance re-ranking without a live Neo4j call
    (uses pre-computed ANIMAL_EXCLUSIVITY from m2).
    """
    from m2_kg_builder import ANIMAL_EXCLUSIVITY

    special = [s.lower() for s in prefs.special_interests]

    for result in fused:
        boost = 0.0
        dest_lower = result.destination.lower()

        # For each special interest, check if the destination is the exclusive home
        for interest in special:
            for animal, exclusive_dests in ANIMAL_EXCLUSIVITY.items():
                if interest in animal.lower():
                    exclusive_lower = [d.lower() for d in exclusive_dests]
                    if dest_lower in exclusive_lower:
                        boost += 0.15   # 1-hop exact match
                    else:
                        boost -= 0.05  # wrong destination for this animal

        # Crowd avoidance boost for known quiet destinations
        quiet_dests = ["zambia", "zimbabwe", "namibia", "botswana"]
        if prefs.avoid_crowds and dest_lower in quiet_dests:
            boost += 0.08

        # Region filter — demote out-of-region destinations
        if prefs.regions:
            region_lower = [r.lower() for r in prefs.regions]
            if result.region.lower() not in region_lower:
                boost -= 0.20

        result.graph_path_boost = round(boost, 4)
        result.fusion_score = round(result.fusion_score + boost, 4)

    fused.sort(key=lambda x: x.fusion_score, reverse=True)
    return fused


# ── Main hybrid retrieval entry point ─────────────────────────────────────────────

def retrieve(prefs: UserPreferences) -> dict:
    """
    Full GraphRAG Fusion pipeline.
    Returns top destinations + all debug metadata for frontend.
    """
    # Step 1: Route query
    plan = classify_query(prefs)
    print(f"\n[Router] {plan.reasoning}")

    # Step 2a: KG retrieval
    kg_results = []
    try:
        kg_results = query_destinations(
            vibes=plan.kg_vibes,
            budget=prefs.budget_tier,
            month=prefs.travel_month,
        )
        print(f"[KG] Retrieved {len(kg_results)} destinations")
    except Exception as e:
        print(f"[KG] Neo4j not available: {e} — falling back to RAG only")
        plan.alpha = 0.0  # adaptive fallback

    # Step 2b: RAG retrieval
    region_filter = prefs.regions[0].lower() if len(prefs.regions) == 1 else None
    rag_results = semantic_search(
        query=plan.rag_query,
        n_results=20,
        region_filter=region_filter,
    )
    print(f"[RAG] Retrieved {len(rag_results)} chunks")

    # Step 3: Fuse
    fused = fuse_results(kg_results, rag_results, plan.alpha, prefs)
    print(f"[Fusion] Merged into {len(fused)} candidates (α={plan.alpha})")

    # Step 4: Re-rank
    fused = apply_graph_path_boost(fused, prefs)
    print(f"[Re-ranker] Applied graph-path distance boosts")

    top = fused[:5]

    return {
        "query_plan":     plan,
        "kg_count":       len(kg_results),
        "rag_count":      len(rag_results),
        "top_destinations": [
            {
                "destination":    r.destination,
                "region":         r.region,
                "fusion_score":   r.fusion_score,
                "kg_score":       r.kg_score,
                "rag_score":      r.rag_score,
                "graph_boost":    r.graph_path_boost,
                "matched_vibes":  r.matched_vibes,
                "highlights":     r.highlights,
                "best_for":       r.best_for,
                "rag_chunks":     r.rag_chunks,
            }
            for r in top
        ],
        "alpha":          plan.alpha,
        "query_type":     plan.query_type,
        "reasoning":      plan.reasoning,
    }


if __name__ == "__main__":
    prefs = UserPreferences(
        vibes=["wildlife", "scenic"],
        budget_tier="mid",
        duration_days=10,
        avoid_crowds=True,
        special_interests=["gorilla"],
        regions=["africa"],
        travel_month="July",
        free_text="I hate crowds and love primates",
    )

    result = retrieve(prefs)
    print(f"\nQuery type: {result['query_type']} | α={result['alpha']}")
    print(f"KG hits: {result['kg_count']} | RAG chunks: {result['rag_count']}")
    print("\nTop destinations:")
    for d in result["top_destinations"]:
        print(f"  {d['destination']:20} score={d['fusion_score']:.3f}  "
              f"kg={d['kg_score']:.3f}  rag={d['rag_score']:.3f}  boost={d['graph_boost']:+.3f}")
