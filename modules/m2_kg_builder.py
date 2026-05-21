"""
Module 2 — Knowledge Graph Builder
Extracts entities and relationships from ingested chunks → Neo4j local.
Schema: Destination, Vibe, Activity, Season, Animal, BudgetTier, ExperienceLevel
"""

import re
import os
from neo4j import GraphDatabase
from m1_pdf_ingestion import ingest_all, DocumentChunk


# ── Neo4j connection ─────────────────────────────────────────────────────────────

NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "tripmind123")


# ── Static knowledge seed data ────────────────────────────────────────────────
# Extracted from both PDFs manually + via entity extraction

DESTINATION_DATA = {
    # Africa — from Africa Safari PDF
    "Kenya": {
        "region": "africa", "country": "Kenya",
        "vibes": ["wildlife", "safari", "scenic", "adventure", "iconic"],
        "best_for": "classic savannah safari",
        "budget_tier": "mid",
        "experience_level": "beginner",
        "best_months": ["July", "August", "September", "January", "February"],
        "avoid_months": ["April", "May"],
        "highlights": ["Masai Mara", "Lake Nakuru", "Amboseli"],
        "animals": ["lion", "wildebeest", "elephant", "rhinoceros", "flamingo"],
    },
    "Tanzania": {
        "region": "africa", "country": "Tanzania",
        "vibes": ["wildlife", "safari", "scenic", "iconic", "adventure"],
        "best_for": "iconic landscapes and well-known safari locations",
        "budget_tier": "mid",
        "experience_level": "beginner",
        "best_months": ["June", "July", "August", "September", "October"],
        "avoid_months": ["April", "May"],
        "highlights": ["Serengeti", "Ngorongoro Crater", "Zanzibar"],
        "animals": ["lion", "elephant", "wildebeest", "cheetah", "leopard"],
    },
    "Uganda": {
        "region": "africa", "country": "Uganda",
        "vibes": ["wildlife", "gorilla", "adventure", "quiet", "unique"],
        "best_for": "gorilla trekking and quieter areas",
        "budget_tier": "mid",
        "experience_level": "intermediate",
        "best_months": ["June", "July", "August", "December", "January", "February"],
        "avoid_months": ["March", "April", "May", "October", "November"],
        "highlights": ["Bwindi Impenetrable Forest", "Queen Elizabeth NP", "Kibale NP", "Murchison Falls"],
        "animals": ["mountain gorilla", "chimpanzee", "lion", "elephant", "leopard"],
    },
    "Rwanda": {
        "region": "africa", "country": "Rwanda",
        "vibes": ["wildlife", "gorilla", "scenic", "easy", "clean"],
        "best_for": "amazing scenery gorilla experiences and easy travel",
        "budget_tier": "luxury",
        "experience_level": "beginner",
        "best_months": ["June", "July", "August", "September"],
        "avoid_months": ["March", "April", "May"],
        "highlights": ["Volcanoes National Park", "Nyungwe National Park", "Kigali"],
        "animals": ["mountain gorilla", "chimpanzee", "golden monkey"],
    },
    "South Africa": {
        "region": "africa", "country": "South Africa",
        "vibes": ["wildlife", "diverse", "coastal", "cultural", "affordable"],
        "best_for": "easy affordable and reliable safari plus varied experiences",
        "budget_tier": "mid",
        "experience_level": "beginner",
        "best_months": ["May", "June", "July", "August", "September"],
        "avoid_months": [],
        "highlights": ["Cape Town", "Kruger National Park", "Garden Route"],
        "animals": ["lion", "elephant", "rhinoceros", "leopard", "buffalo", "penguin", "whale"],
    },
    "Namibia": {
        "region": "africa", "country": "Namibia",
        "vibes": ["desert", "isolation", "scenic", "unique", "adventure"],
        "best_for": "isolation amongst stunning desert landscapes",
        "budget_tier": "mid",
        "experience_level": "intermediate",
        "best_months": ["May", "June", "July", "August", "September", "October"],
        "avoid_months": ["December", "January", "February"],
        "highlights": ["Sossusvlei", "Etosha National Park", "Fish River Canyon"],
        "animals": ["meerkat", "oryx", "lion", "elephant", "cheetah", "rhino"],
    },
    "Botswana": {
        "region": "africa", "country": "Botswana",
        "vibes": ["wildlife", "uncrowded", "luxury", "isolation", "pristine"],
        "best_for": "limited crowds and best high-end safari experiences",
        "budget_tier": "luxury",
        "experience_level": "intermediate",
        "best_months": ["May", "June", "July", "August", "September", "October"],
        "avoid_months": ["January", "February"],
        "highlights": ["Okavango Delta", "Chobe National Park", "Moremi Game Reserve"],
        "animals": ["elephant", "lion", "leopard", "wild dog", "hippo", "crocodile"],
    },
    "Zambia": {
        "region": "africa", "country": "Zambia",
        "vibes": ["wildlife", "isolation", "unique", "uncrowded", "adventure"],
        "best_for": "isolated world-class safari for experienced travellers",
        "budget_tier": "luxury",
        "experience_level": "experienced",
        "best_months": ["May", "June", "July", "August", "September", "October"],
        "avoid_months": ["November", "December", "January", "February", "March", "April"],
        "highlights": ["Victoria Falls", "Lower Zambezi NP", "South Luangwa NP"],
        "animals": ["elephant", "lion", "leopard", "hippo", "crocodile", "tiger fish"],
    },
    "Zimbabwe": {
        "region": "africa", "country": "Zimbabwe",
        "vibes": ["wildlife", "uncrowded", "adventure", "unique", "historic"],
        "best_for": "east and southern Africa style safaris in quiet areas",
        "budget_tier": "mid",
        "experience_level": "intermediate",
        "best_months": ["April", "May", "June", "July", "August", "September", "October"],
        "avoid_months": [],
        "highlights": ["Victoria Falls", "Hwange National Park", "Mana Pools NP"],
        "animals": ["elephant", "lion", "leopard", "hyena", "cheetah", "wild dog"],
    },
    # Europe — from Europe Travel PDF
    "Rome": {
        "region": "europe", "country": "Italy",
        "vibes": ["cultural", "history", "iconic", "food", "scenic"],
        "best_for": "history culture and culinary delights",
        "budget_tier": "mid",
        "experience_level": "beginner",
        "best_months": ["April", "May", "September", "October"],
        "avoid_months": ["July", "August"],
        "highlights": ["Colosseum", "Vatican", "Trastevere", "Roman Forum"],
        "animals": [],
    },
    "Bucharest": {
        "region": "europe", "country": "Romania",
        "vibes": ["cultural", "nightlife", "history", "budget", "hidden gem"],
        "best_for": "architecture nightlife and budget travel",
        "budget_tier": "budget",
        "experience_level": "beginner",
        "best_months": ["May", "June", "September", "October"],
        "avoid_months": ["January", "February"],
        "highlights": ["Old Town Lipscani", "Palace of Parliament", "Herastrau Park"],
        "animals": [],
    },
    "Pula": {
        "region": "europe", "country": "Croatia",
        "vibes": ["coastal", "history", "scenic", "hidden gem", "beach"],
        "best_for": "ancient roman history and coastal beauty",
        "budget_tier": "mid",
        "experience_level": "beginner",
        "best_months": ["May", "June", "September"],
        "avoid_months": ["July", "August"],
        "highlights": ["Pula Arena", "Kamenjak Peninsula", "Brijuni Islands"],
        "animals": [],
    },
}

ANIMAL_EXCLUSIVITY = {
    "mountain gorilla": ["Uganda", "Rwanda"],
    "chimpanzee":       ["Uganda", "Rwanda", "Tanzania"],
    "meerkat":          ["Namibia", "Botswana"],
    "penguin":          ["South Africa"],
    "whale":            ["South Africa"],
    "flamingo":         ["Kenya"],
}


# ── Schema setup ─────────────────────────────────────────────────────────────────

SCHEMA_QUERIES = [
    "CREATE CONSTRAINT dest_name IF NOT EXISTS FOR (d:Destination) REQUIRE d.name IS UNIQUE",
    "CREATE CONSTRAINT vibe_name IF NOT EXISTS FOR (v:Vibe) REQUIRE v.name IS UNIQUE",
    "CREATE CONSTRAINT animal_name IF NOT EXISTS FOR (a:Animal) REQUIRE a.name IS UNIQUE",
    "CREATE CONSTRAINT month_name IF NOT EXISTS FOR (m:Month) REQUIRE m.name IS UNIQUE",
    "CREATE CONSTRAINT budget_name IF NOT EXISTS FOR (b:BudgetTier) REQUIRE b.name IS UNIQUE",
    "CREATE CONSTRAINT exp_name IF NOT EXISTS FOR (e:ExperienceLevel) REQUIRE e.name IS UNIQUE",
    "CREATE CONSTRAINT highlight_name IF NOT EXISTS FOR (h:Highlight) REQUIRE h.name IS UNIQUE",
]


# ── Graph write transactions ──────────────────────────────────────────────────────

def create_destination(tx, name: str, data: dict):
    tx.run("""
        MERGE (d:Destination {name: $name})
        SET d.region = $region,
            d.country = $country,
            d.best_for = $best_for,
            d.experience_level = $exp
        WITH d
        MERGE (b:BudgetTier {name: $budget})
        MERGE (d)-[:SUITS_BUDGET]->(b)
        WITH d
        MERGE (e:ExperienceLevel {name: $exp})
        MERGE (d)-[:SUITS_TRAVELLER]->(e)
    """, name=name, region=data["region"], country=data["country"],
         best_for=data["best_for"], budget=data["budget_tier"],
         exp=data["experience_level"])


def create_vibes(tx, dest_name: str, vibes: list):
    for vibe in vibes:
        tx.run("""
            MERGE (d:Destination {name: $dest})
            MERGE (v:Vibe {name: $vibe})
            MERGE (d)-[:HAS_VIBE]->(v)
        """, dest=dest_name, vibe=vibe)


def create_seasons(tx, dest_name: str, best_months: list, avoid_months: list):
    for month in best_months:
        tx.run("""
            MERGE (d:Destination {name: $dest})
            MERGE (m:Month {name: $month})
            MERGE (d)-[:BEST_IN]->(m)
        """, dest=dest_name, month=month)
    for month in avoid_months:
        tx.run("""
            MERGE (d:Destination {name: $dest})
            MERGE (m:Month {name: $month})
            MERGE (d)-[:AVOID_IN]->(m)
        """, dest=dest_name, month=month)


def create_animals(tx, dest_name: str, animals: list):
    for animal in animals:
        tx.run("""
            MERGE (d:Destination {name: $dest})
            MERGE (a:Animal {name: $animal})
            MERGE (d)-[:BEST_FOR_SEEING]->(a)
        """, dest=dest_name, animal=animal)


def create_animal_exclusivity(tx):
    for animal, destinations in ANIMAL_EXCLUSIVITY.items():
        for dest in destinations:
            tx.run("""
                MERGE (a:Animal {name: $animal})
                MERGE (d:Destination {name: $dest})
                MERGE (a)-[:FOUND_ONLY_IN]->(d)
            """, animal=animal, dest=dest)


def create_highlights(tx, dest_name: str, highlights: list):
    for h in highlights:
        tx.run("""
            MERGE (d:Destination {name: $dest})
            MERGE (h:Highlight {name: $highlight})
            MERGE (d)-[:HAS_HIGHLIGHT]->(h)
            SET h.destination = $dest
        """, dest=dest_name, highlight=h)


def create_similarity_edges(tx):
    """Connect destinations that share vibes as SIMILAR_TO."""
    tx.run("""
        MATCH (d1:Destination)-[:HAS_VIBE]->(v:Vibe)<-[:HAS_VIBE]-(d2:Destination)
        WHERE d1.name < d2.name
        WITH d1, d2, count(v) AS shared
        WHERE shared >= 2
        MERGE (d1)-[r:SIMILAR_TO]-(d2)
        SET r.shared_vibes = shared
    """)


# ── Main entry ───────────────────────────────────────────────────────────────────

def build_knowledge_graph(neo4j_uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD):
    driver = GraphDatabase.driver(neo4j_uri, auth=(user, password))

    print("Setting up Neo4j schema constraints...")
    with driver.session() as session:
        for q in SCHEMA_QUERIES:
            try:
                session.run(q)
            except Exception:
                pass  # constraint may already exist

    print("Populating destination nodes...")
    with driver.session() as session:
        for dest_name, data in DESTINATION_DATA.items():
            session.execute_write(create_destination, dest_name, data)
            session.execute_write(create_vibes, dest_name, data["vibes"])
            session.execute_write(create_seasons, dest_name, data["best_months"], data["avoid_months"])
            session.execute_write(create_animals, dest_name, data["animals"])
            session.execute_write(create_highlights, dest_name, data["highlights"])
            print(f"  + {dest_name}")

        print("Creating animal exclusivity edges...")
        session.execute_write(create_animal_exclusivity)

        print("Creating similarity edges...")
        session.execute_write(create_similarity_edges)

    driver.close()
    print(f"\nKnowledge graph built: {len(DESTINATION_DATA)} destinations loaded into Neo4j")


def query_destinations(vibes: list, budget: str = None, month: str = None,
                        animals: list = None, avoid_crowds: bool = False,
                        neo4j_uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD) -> list[dict]:
    """
    Query Neo4j for destinations matching user preferences.
    Returns list of dicts with name, score, matched_vibes, highlights.
    """
    driver = GraphDatabase.driver(neo4j_uri, auth=(user, password))

    results = []
    with driver.session() as session:
        # Build dynamic Cypher based on filters
        vibe_filter = " OR ".join([f"v.name = '{v}'" for v in vibes]) if vibes else "true"

        cypher = f"""
            MATCH (d:Destination)-[:HAS_VIBE]->(v:Vibe)
            WHERE {vibe_filter}
            WITH d, collect(v.name) AS matched_vibes, count(v) AS vibe_score
        """

        if budget:
            cypher += f"""
                MATCH (d)-[:SUITS_BUDGET]->(b:BudgetTier {{name: '{budget}'}})
            """

        if month:
            cypher += f"""
                OPTIONAL MATCH (d)-[r:BEST_IN]->(m:Month {{name: '{month}'}})
                WITH d, matched_vibes, vibe_score, count(m) AS season_score
            """
        else:
            cypher += " WITH d, matched_vibes, vibe_score, 0 AS season_score"

        cypher += """
            OPTIONAL MATCH (d)-[:HAS_HIGHLIGHT]->(h:Highlight)
            WITH d, matched_vibes, vibe_score, season_score, collect(h.name) AS highlights
            RETURN d.name AS name, d.region AS region, d.best_for AS best_for,
                   matched_vibes, vibe_score, season_score, highlights,
                   (vibe_score * 2 + season_score) AS total_score
            ORDER BY total_score DESC
            LIMIT 8
        """

        rows = session.run(cypher)
        for row in rows:
            results.append({
                "name": row["name"],
                "region": row["region"],
                "best_for": row["best_for"],
                "matched_vibes": row["matched_vibes"],
                "vibe_score": row["vibe_score"],
                "highlights": row["highlights"],
                "total_score": row["total_score"],
            })

    driver.close()
    return results


if __name__ == "__main__":
    build_knowledge_graph()

    print("\nTest query: wildlife + quiet, mid budget, July...")
    results = query_destinations(
        vibes=["wildlife", "quiet", "isolation"],
        budget="mid",
        month="July",
    )
    for r in results:
        print(f"  {r['name']} — score={r['total_score']} vibes={r['matched_vibes']}")
