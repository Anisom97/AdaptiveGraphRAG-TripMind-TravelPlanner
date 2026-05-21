"""
Module 7 — Streamlit Frontend
Full itinerary planner UI with preference inputs, fusion debug panel,
and day-by-day itinerary cards.
Run: streamlit run frontend/app.py
"""

import streamlit as st
import requests
import json
import time

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="TripMind — AI Itinerary Planner",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; max-width: 1200px; }
    .stButton > button { width: 100%; border-radius: 10px; font-weight: 600; padding: 0.6rem 1rem; }
    .day-card { background: #f8f9fa; border-radius: 12px; padding: 1.2rem 1.4rem;
                margin-bottom: 0.8rem; border-left: 4px solid #4A90E2; }
    .day-header { font-size: 1.05rem; font-weight: 700; margin-bottom: 0.5rem; color: #1a1a2e; }
    .activity-row { padding: 0.35rem 0; border-bottom: 1px solid #e9ecef; font-size: 0.9rem; }
    .activity-row:last-child { border-bottom: none; }
    .time-badge { color: #6c757d; font-size: 0.8rem; min-width: 90px; display: inline-block; }
    .highlight-tag { background: #fff3cd; color: #856404; font-size: 0.75rem;
                     padding: 2px 8px; border-radius: 8px; margin-left: 6px; }
    .source-tag { background: #e8f4f8; color: #0c6475; font-size: 0.72rem;
                  padding: 2px 7px; border-radius: 6px; margin-left: 4px; }
    .vibe-chip { background: #e9ecef; color: #495057; font-size: 0.75rem;
                 padding: 2px 9px; border-radius: 10px; margin: 2px; display: inline-block; }
    .fusion-box { background: #f0f4ff; border-radius: 10px; padding: 1rem 1.2rem;
                  margin-bottom: 1rem; border: 1px solid #d4ddff; }
    .stat-number { font-size: 2rem; font-weight: 700; color: #1a1a2e; line-height: 1; }
    .stat-label  { font-size: 0.78rem; color: #6c757d; margin-top: 2px; }
    .header-brand { font-size: 1.6rem; font-weight: 800; color: #1a1a2e; }
    .header-sub   { font-size: 0.85rem; color: #6c757d; margin-top: -4px; }
    .status-dot   { width: 8px; height: 8px; border-radius: 50%;
                    display: inline-block; margin-right: 5px; }
    .dot-green { background: #28a745; }
    .dot-red   { background: #dc3545; }
    .dot-amber { background: #ffc107; }
    .section-divider { border: none; border-top: 1px solid #e9ecef; margin: 1rem 0; }
    div[data-testid="metric-container"] { background: #f8f9fa; border-radius: 10px; padding: 0.8rem; }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────────

col_logo, col_title, col_status = st.columns([1, 6, 3])
with col_logo:
    st.markdown("### ✦")
with col_title:
    st.markdown('<div class="header-brand">TripMind</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-sub">Hybrid KG + RAG · AI itinerary planner · MTech thesis demo</div>', unsafe_allow_html=True)
with col_status:
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        neo4j_ok  = health.get("neo4j", "") == "ok"
        chroma_ok = health.get("vector_store", "") == "ok"
        ollama_ok = health.get("ollama", "") == "ok"
        n_docs    = health.get("total_documents", 0)

        neo4j_dot  = "dot-green" if neo4j_ok  else "dot-red"
        chroma_dot = "dot-green" if chroma_ok else "dot-red"
        ollama_dot = "dot-green" if ollama_ok else "dot-amber"

        st.markdown(f"""
        <div style="font-size:0.78rem;line-height:2;padding-top:0.5rem;">
            <span class="status-dot {chroma_dot}"></span>ChromaDB ({n_docs} docs)<br>
            <span class="status-dot {neo4j_dot}"></span>Neo4j KG<br>
            <span class="status-dot {ollama_dot}"></span>Ollama LLM
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        st.markdown('<div style="font-size:0.78rem;color:#dc3545;">⚠ API not running<br>Start with: python m6_api.py</div>', unsafe_allow_html=True)

st.divider()

# ── Main layout: sidebar + content ───────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Preferences")

    # Region
    st.markdown("**Destination region**")
    regions_all = ["Africa", "Europe", "Asia", "Americas"]
    selected_regions = []
    cols = st.columns(2)
    defaults_region = {"Africa": True, "Europe": True, "Asia": False, "Americas": False}
    for i, r in enumerate(regions_all):
        if cols[i % 2].checkbox(r, value=defaults_region.get(r, False), key=f"reg_{r}"):
            selected_regions.append(r.lower())

    st.markdown("---")

    # Vibes
    st.markdown("**Travel vibe** *(pick all that apply)*")
    ALL_VIBES = [
        "Wildlife", "Beach", "Adventure", "Scenic", "Nightlife",
        "Cultural", "Isolation", "Luxury", "History", "Food",
        "Gorilla", "Desert", "Coastal",
    ]
    default_vibes = {"Wildlife", "Scenic"}
    selected_vibes = []
    vibe_cols = st.columns(2)
    for i, v in enumerate(ALL_VIBES):
        if vibe_cols[i % 2].checkbox(v, value=(v in default_vibes), key=f"vibe_{v}"):
            selected_vibes.append(v.lower())

    st.markdown("---")

    # Budget
    st.markdown("**Budget tier**")
    budget = st.radio(
        "", ["Budget", "Mid-range", "Luxury"],
        index=1, horizontal=True, label_visibility="collapsed"
    ).lower().replace("-range", "").replace("mid", "mid")
    budget_map = {"budget": "budget", "mid": "mid", "luxury": "luxury"}
    budget_key = "mid" if "mid" in budget else budget

    st.markdown("---")

    # Duration
    duration = st.slider("**Duration (days)**", 3, 21, 10, step=1)

    # Month
    months = ["Flexible", "January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    travel_month = st.selectbox("**Travel month**", months)
    travel_month = None if travel_month == "Flexible" else travel_month

    # Toggles
    avoid_crowds = st.toggle("Avoid crowds / quiet destinations", value=True)

    st.markdown("---")

    # Special interests
    st.markdown("**Special interests**")
    ALL_INTERESTS = ["Gorillas", "Big Five", "Walking Safari", "Hiking", "Cuisine",
                     "Nightlife", "History", "Beaches", "Snorkelling"]
    default_interests = {"Gorillas"}
    selected_interests = []
    int_cols = st.columns(2)
    for i, interest in enumerate(ALL_INTERESTS):
        if int_cols[i % 2].checkbox(interest, value=(interest in default_interests), key=f"int_{interest}"):
            selected_interests.append(interest.lower())

    st.markdown("---")

    # Free text
    free_text = st.text_area(
        "**Free-text prompt**",
        placeholder="e.g. I hate crowds, love primates, want something off the beaten track...",
        height=90,
    )

    st.markdown("---")

    generate_btn = st.button("✦ Generate Itinerary", type="primary", use_container_width=True)


# ── Main content area ─────────────────────────────────────────────────────────────

if "itinerary" not in st.session_state:
    st.session_state.itinerary = None
if "generating" not in st.session_state:
    st.session_state.generating = False

if generate_btn:
    st.session_state.generating = True
    payload = {
        "vibes":             selected_vibes,
        "budget_tier":       budget_key,
        "duration_days":     duration,
        "avoid_crowds":      avoid_crowds,
        "special_interests": selected_interests,
        "regions":           selected_regions,
        "travel_month":      travel_month,
        "free_text":         free_text,
    }

    with st.spinner("Routing query · fusing KG + RAG · composing with Mistral..."):
        try:
            resp = requests.post(f"{API_URL}/generate", json=payload, timeout=180)
            resp.raise_for_status()
            st.session_state.itinerary = resp.json()
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Start it with: `cd modules && python m6_api.py`")
            st.session_state.itinerary = None
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.session_state.itinerary = None
    st.session_state.generating = False


# ── Display itinerary ─────────────────────────────────────────────────────────────

itin = st.session_state.itinerary

if itin is None:
    # Welcome state
    st.markdown("### Set your preferences and hit **✦ Generate Itinerary**")
    st.markdown("""
    **How TripMind works:**
    1. Your preferences are parsed and classified by the **Query Router**
    2. The **Knowledge Graph** (Neo4j) retrieves structurally matched destinations
    3. **ChromaDB** retrieves semantically relevant travel guide chunks
    4. The **Fusion Scorer** merges both with weighted α·KG + (1-α)·RAG
    5. The **Re-ranker** boosts destinations by graph-path distance
    6. **Mistral-7B** (via Ollama) composes your personalised itinerary

    *Data sources: Africa Safari Guide (Explorer Society) + Europe Travel Guide (Wandering Lewis)*
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Destinations in KG", "12")
    col2.metric("RAG chunks", "—")
    col3.metric("Relationships", "90+")

elif "error" in itin:
    st.error(itin["error"])

else:
    # ── Itinerary header
    meta = itin.get("_retrieval_meta", {})
    alpha = meta.get("alpha", 0.5)
    kg_count  = meta.get("kg_count", 0)
    rag_count = meta.get("rag_count", 0)

    title    = itin.get("title", "Your itinerary")
    subtitle = itin.get("subtitle", "")

    st.markdown(f"## {title}")
    st.markdown(f"*{subtitle}*")

    # ── Stats row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration", f"{itin.get('total_days', duration)} days")
    c2.metric("KG matches", kg_count)
    c3.metric("RAG chunks", rag_count)
    c4.metric("Query type", meta.get("query_type", "hybrid").title())

    st.divider()

    # ── Fusion debug panel
    with st.expander("GraphRAG Fusion debug panel", expanded=True):
        st.markdown(f"**Query router reasoning:** {meta.get('reasoning', '—')}")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Knowledge Graph weight (α)**")
            st.progress(alpha, text=f"{alpha:.0%} KG")
        with col_b:
            st.markdown("**RAG weight (1-α)**")
            st.progress(1 - alpha, text=f"{1-alpha:.0%} RAG")

        top_dests = meta.get("top_destinations", [])
        if top_dests:
            st.markdown("**Destination fusion scores**")
            for d in top_dests:
                st.markdown(
                    f"`{d['destination']:20}` "
                    f"fusion=**{d['fusion_score']:.3f}** "
                    f"kg={d['kg_score']:.3f} "
                    f"rag={d['rag_score']:.3f} "
                    f"boost={d['graph_boost']:+.3f}"
                )

    st.divider()

    # ── Day cards
    st.markdown("### Day-by-day itinerary")
    days = itin.get("days", [])

    for day in days:
        day_range = day.get("day_range", "")
        location  = day.get("location", "")
        theme     = day.get("theme", "")
        activities = day.get("activities", [])
        tags       = day.get("tags", [])
        accom_tip  = day.get("accommodation_tip", "")

        with st.container():
            st.markdown(f"""
            <div class="day-card">
                <div class="day-header">
                    <span style="background:#e8f0fe;color:#1967d2;padding:2px 10px;border-radius:10px;font-size:0.82rem;margin-right:8px;">{day_range}</span>
                    {location} — {theme}
                </div>
            """, unsafe_allow_html=True)

            for act in activities:
                time_label = act.get("time", "")
                desc       = act.get("description", "")
                is_hl      = act.get("highlight", False)
                source     = act.get("source_hint", "")
                hl_badge   = '<span class="highlight-tag">Highlight</span>' if is_hl else ""
                src_badge  = f'<span class="source-tag">{source[:50]}</span>' if source else ""
                st.markdown(f"""
                <div class="activity-row">
                    <span class="time-badge">{time_label}</span>
                    {desc}{hl_badge}{src_badge}
                </div>
                """, unsafe_allow_html=True)

            # Tags + accommodation
            tags_html = " ".join(f'<span class="vibe-chip">{t}</span>' for t in tags)
            st.markdown(f'<div style="margin-top:0.5rem;">{tags_html}</div>', unsafe_allow_html=True)
            if accom_tip:
                st.markdown(f'<div style="font-size:0.8rem;color:#6c757d;margin-top:0.4rem;">🛏 {accom_tip}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Practical tips
    tips = itin.get("practical_tips", [])
    if tips:
        st.divider()
        st.markdown("### Practical tips")
        for tip in tips:
            st.markdown(f"- {tip}")

    # ── Budget + best months
    col_b, col_m = st.columns(2)
    with col_b:
        budget_est = itin.get("estimated_budget", "")
        if budget_est:
            st.info(f"**Estimated budget:** {budget_est}")
    with col_m:
        best_months = itin.get("best_months", [])
        if best_months:
            st.success(f"**Best travel months:** {', '.join(best_months)}")

    # ── Refinement actions
    st.divider()
    st.markdown("### Refine your itinerary")
    ref_cols = st.columns(3)
    if ref_cols[0].button("Add a beach extension"):
        st.info("Tip: Add 'Beach' to your vibes and increase duration by 3–4 days, then regenerate.")
    if ref_cols[1].button("Make it more budget-friendly"):
        st.info("Tip: Switch budget tier to 'Budget' and regenerate.")
    if ref_cols[2].button("Export as JSON"):
        clean = {k: v for k, v in itin.items() if not k.startswith("_")}
        st.download_button(
            "Download itinerary.json",
            data=json.dumps(clean, indent=2),
            file_name="itinerary.json",
            mime="application/json",
        )

    # ── Raw JSON toggle
    with st.expander("Raw JSON output"):
        st.json(itin)
