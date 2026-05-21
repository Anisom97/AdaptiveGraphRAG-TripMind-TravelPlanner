"""
Module 5 — LLM Composer
Takes top fused destinations + RAG context chunks and generates
a structured day-by-day itinerary using Ollama Mistral-7B locally.
Falls back to a template-based generator if Ollama is unavailable.
"""

import os
import json
import requests
from m4_graphrag_fusion import UserPreferences, retrieve


OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")


# ── Prompt builder ────────────────────────────────────────────────────────────────

def build_system_prompt() -> str:
    return """You are TripMind, an expert AI travel planner.
You create personalised, day-by-day travel itineraries grounded in real travel guide content.
Always structure your response as a valid JSON object.
Never hallucinate facts — only use information from the provided context chunks.
Be specific about places, activities, and timings."""


def build_user_prompt(prefs: UserPreferences, retrieval: dict) -> str:
    top = retrieval["top_destinations"]
    if not top:
        return "No destinations found. Please broaden your preferences."

    # Build context from RAG chunks
    context_blocks = []
    for dest_data in top[:3]:
        dest = dest_data["destination"]
        chunks = dest_data["rag_chunks"]
        if chunks:
            combined = " ".join(c["text"] for c in chunks[:2])[:600]
            context_blocks.append(f"[{dest}]: {combined}")

    context_str = "\n\n".join(context_blocks)

    # Build top destinations summary
    top_names = [d["destination"] for d in top[:3]]
    top_summary = ", ".join(top_names)
    highlights = []
    for d in top[:3]:
        if d["highlights"]:
            highlights.extend(d["highlights"][:2])

    prompt = f"""
Create a {prefs.duration_days}-day travel itinerary for a traveller with these preferences:
- Regions: {", ".join(prefs.regions) if prefs.regions else "any"}
- Travel vibes: {", ".join(prefs.vibes)}
- Budget: {prefs.budget_tier}
- Avoid crowds: {prefs.avoid_crowds}
- Special interests: {", ".join(prefs.special_interests) if prefs.special_interests else "none"}
- Travel month: {prefs.travel_month or "flexible"}
- Additional request: "{prefs.free_text}"

Top matched destinations from hybrid retrieval: {top_summary}
Key highlights available: {", ".join(highlights[:6])}

Context from travel guides:
{context_str}

Return a JSON object with this exact structure:
{{
  "title": "Short evocative itinerary title",
  "subtitle": "Destinations covered — duration — style",
  "total_days": {prefs.duration_days},
  "days": [
    {{
      "day_range": "Day 1-2",
      "location": "City/Park name",
      "theme": "Short theme e.g. Arrival and city exploration",
      "activities": [
        {{
          "time": "Morning/Afternoon/Evening/Full day",
          "description": "What to do — specific and detailed",
          "highlight": true or false,
          "source_hint": "Brief reference to guide source e.g. Africa Safari Guide"
        }}
      ],
      "tags": ["tag1", "tag2"],
      "accommodation_tip": "One short tip on where to stay"
    }}
  ],
  "practical_tips": ["tip1", "tip2", "tip3"],
  "best_months": ["Month1", "Month2"],
  "estimated_budget": "e.g. $150-250/day mid-range"
}}

Generate {min(prefs.duration_days // 2 + 1, 6)} day blocks covering {prefs.duration_days} days total.
"""
    return prompt.strip()


# ── Ollama inference ──────────────────────────────────────────────────────────────

def call_ollama(system_prompt: str, user_prompt: str) -> str:
    """Call local Ollama API. Returns raw text response."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 2000,
        },
    }
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def extract_json(raw: str) -> dict:
    """Extract JSON from LLM response, stripping markdown fences."""
    raw = raw.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find JSON block inside the response
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


import re

# ── Template fallback ──────────────────────────────────────────────────────────────

def template_itinerary(prefs: UserPreferences, retrieval: dict) -> dict:
    """
    Rule-based itinerary generator used when Ollama is unavailable.
    Produces a structured itinerary from KG highlights + RAG chunks.
    """
    top = retrieval["top_destinations"]
    if not top:
        return {"error": "No destinations found"}

    days = prefs.duration_days
    day_blocks = []

    # Distribute days across top 3 destinations
    dest_allocs = []
    if len(top) >= 3:
        allocs = [days // 3, days // 3, days - 2 * (days // 3)]
    elif len(top) == 2:
        allocs = [days // 2, days - days // 2]
    else:
        allocs = [days]

    for i, dest_data in enumerate(top[:len(allocs)]):
        dest   = dest_data["destination"]
        alloc  = allocs[i]
        chunks = dest_data["rag_chunks"]
        highlights = dest_data["highlights"][:3]

        # Extract key sentences from RAG chunks
        descriptions = []
        for c in chunks[:2]:
            sentences = [s.strip() for s in c["text"].split(".") if len(s.strip()) > 40]
            descriptions.extend(sentences[:2])

        activities = []
        if i == 0:
            activities.append({
                "time": "Afternoon",
                "description": f"Arrive in {dest} · check in · orientation walk",
                "highlight": False,
                "source_hint": dest_data.get("best_for", ""),
            })
        if highlights:
            activities.append({
                "time": "Full day",
                "description": f"Explore {highlights[0]}" + (f" and {highlights[1]}" if len(highlights) > 1 else ""),
                "highlight": True,
                "source_hint": f"Best for: {dest_data['best_for'][:60]}",
            })
        if descriptions:
            activities.append({
                "time": "Day 2+",
                "description": descriptions[0][:120] if descriptions else f"Discover {dest}",
                "highlight": False,
                "source_hint": dest_data["rag_chunks"][0]["source_title"] if dest_data["rag_chunks"] else "",
            })

        start = sum(allocs[:i]) + 1
        end   = start + alloc - 1
        day_range = f"Day {start}" if start == end else f"Day {start}–{end}"

        day_blocks.append({
            "day_range":         day_range,
            "location":          dest,
            "theme":             dest_data["best_for"][:60] if dest_data["best_for"] else dest,
            "activities":        activities,
            "tags":              dest_data["matched_vibes"][:3],
            "accommodation_tip": f"Mid-range lodge or guesthouse in {dest}",
        })

    top_names = " + ".join(d["destination"] for d in top[:len(allocs)])
    return {
        "title":            f"{top_names} — {days} Days",
        "subtitle":         f"{top_names} · {days} days · {prefs.budget_tier.title()} budget",
        "total_days":       days,
        "days":             day_blocks,
        "practical_tips":   [
            "Book gorilla permits months in advance if applicable",
            "Pack lightweight layers — mornings can be cool",
            "Carry USD cash in smaller denominations for rural areas",
        ],
        "best_months":      [prefs.travel_month] if prefs.travel_month else ["July", "August"],
        "estimated_budget": "$100–200/day mid-range all-inclusive",
        "_generated_by":    "template_fallback",
    }


# ── Main compose function ─────────────────────────────────────────────────────────

def compose_itinerary(prefs: UserPreferences) -> dict:
    """
    Full pipeline: retrieve → build prompt → call LLM → parse JSON.
    Falls back to template if Ollama is unavailable.
    """
    # Retrieval
    retrieval = retrieve(prefs)

    # Try Ollama
    ollama_available = False
    try:
        health = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        ollama_available = health.status_code == 200
    except Exception:
        pass

    itinerary = {}

    if ollama_available:
        print(f"[LLM] Using Ollama ({OLLAMA_MODEL})...")
        system_prompt = build_system_prompt()
        user_prompt   = build_user_prompt(prefs, retrieval)
        try:
            raw = call_ollama(system_prompt, user_prompt)
            itinerary = extract_json(raw)
            itinerary["_generated_by"] = f"ollama:{OLLAMA_MODEL}"
        except Exception as e:
            print(f"[LLM] Ollama failed ({e}), falling back to template")
            itinerary = template_itinerary(prefs, retrieval)
    else:
        print("[LLM] Ollama not running — using template fallback")
        itinerary = template_itinerary(prefs, retrieval)

    # Attach retrieval metadata for frontend debug panel
    itinerary["_retrieval_meta"] = {
        "kg_count":   retrieval["kg_count"],
        "rag_count":  retrieval["rag_count"],
        "alpha":      retrieval["alpha"],
        "query_type": retrieval["query_type"],
        "reasoning":  retrieval["reasoning"],
        "top_destinations": [
            {
                "destination": d["destination"],
                "fusion_score": d["fusion_score"],
                "kg_score": d["kg_score"],
                "rag_score": d["rag_score"],
                "graph_boost": d["graph_boost"],
            }
            for d in retrieval["top_destinations"]
        ],
    }

    return itinerary


if __name__ == "__main__":
    prefs = UserPreferences(
        vibes=["wildlife", "scenic"],
        budget_tier="mid",
        duration_days=10,
        avoid_crowds=True,
        special_interests=["gorilla"],
        regions=["africa"],
        travel_month="July",
        free_text="I love primates and hate crowds",
    )

    result = compose_itinerary(prefs)
    print(json.dumps(result, indent=2)[:2000])
