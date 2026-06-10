#!/usr/bin/env python3
"""Smoke test: build search data for a handful of icons, then check quality.

Run with:
    GEMINI_API_KEY=... uv run python scripts/test_search_quality.py

Uses ~10 icons chosen as positive/negative controls for specific queries.
"""

import json
import os
import pathlib
import sqlite3
import sys
import tempfile

# Test icons: chosen so we can write queries with clear expected matches
TEST_ICONS = [
    "heart",
    "credit-card",
    "bird",
    "shield-check",
    "hammer",
    "tree-pine",
    "music",
    "mail",
    "skull",
    "sun",
]

# (query, expected_top_3, should_NOT_be_top_3)
QUALITY_CHECKS = [
    ("love", ["heart"], ["skull", "hammer"]),
    ("payment", ["credit-card"], ["bird", "tree-pine"]),
    ("nature wildlife", ["bird", "tree-pine"], ["credit-card", "mail"]),
    ("security", ["shield-check"], ["music", "sun"]),
    ("manual labor construction", ["hammer"], ["music", "bird"]),
    ("email message", ["mail"], ["hammer", "skull"]),
    ("death danger", ["skull"], ["heart", "sun"]),
    ("weather sunny", ["sun"], ["skull", "mail"]),
]


def main() -> int:
    if not os.environ.get("GEMINI_API_KEY"):
        print("Set GEMINI_API_KEY to run this test.")
        return 1

    main_db = pathlib.Path("src/lucide/data/lucide-icons.db")
    if not main_db.exists():
        print(f"Icons DB not found: {main_db}")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)

        # Build a small main DB with just our test icons
        small_db = tmp_path / "small-icons.db"
        conn_src = sqlite3.connect(f"file:{main_db}?mode=ro", uri=True)
        conn_dst = sqlite3.connect(small_db)
        conn_dst.execute(
            "CREATE TABLE icons (name TEXT PRIMARY KEY, svg TEXT NOT NULL)"
        )
        conn_dst.execute(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        conn_dst.execute("INSERT INTO metadata VALUES ('version', 'test')")

        for name in TEST_ICONS:
            row = conn_src.execute(
                "SELECT svg FROM icons WHERE name = ?", (name,)
            ).fetchone()
            if row:
                conn_dst.execute("INSERT INTO icons VALUES (?, ?)", (name, row[0]))
            else:
                print(f"  WARNING: icon '{name}' not in DB, skipping")

        conn_dst.commit()
        conn_dst.close()
        conn_src.close()

        icon_count = len(TEST_ICONS)
        print(f"Built small DB with {icon_count} icons\n")

        # Generate descriptions via VLM
        search_db = tmp_path / "test-search.db"

        from lucide.build_search import build_search_db, generate_descriptions

        print("Generating descriptions + embeddings...")
        jsonl = tmp_path / "test-descriptions.jsonl"
        generate_descriptions(small_db, jsonl, verbose=True)

        # Create a minimal clusters file
        clusters = {"clusters": {"-1": {"icons": [], "theme": "Unclustered"}}}
        clusters_path = tmp_path / "clusters.json"
        clusters_path.write_text(json.dumps(clusters))

        build_search_db(jsonl, search_db, clusters_path, verbose=True)

        # Show descriptions
        conn = sqlite3.connect(f"file:{search_db}?mode=ro", uri=True)
        print("\n--- VLM Descriptions ---")
        for row in conn.execute(
            "SELECT name, description FROM icon_descriptions ORDER BY name"
        ):
            print(f"\n  {row[0]}:")
            print(f"    {row[1]}")

        # Run search quality checks
        print("\n\n--- Search Quality Checks ---\n")

        # Point search at our test DB
        os.environ["LUCIDE_SEARCH_DB_PATH"] = str(search_db)

        # Need to reset the module caches
        from lucide import search

        search._search_index = None
        search._embedder_instance = None

        # Monkey-patch version check to return our test version
        search._get_icons_version = lambda: "test"

        passed = 0
        failed = 0

        for query, expected_hits, expected_misses in QUALITY_CHECKS:
            results = search.search_icons(query, limit=3)
            top_names = [r.name for r in results]
            top_scores = {r.name: f"{r.score:.3f}" for r in results}

            # Check: at least one expected hit is in top 3
            hits_found = [e for e in expected_hits if e in top_names]
            misses_found = [m for m in expected_misses if m in top_names]

            ok = len(hits_found) > 0 and len(misses_found) == 0
            status = "PASS" if ok else "FAIL"

            if ok:
                passed += 1
            else:
                failed += 1

            print(f"  [{status}] query={query!r}")
            scores_str = ", ".join(f"{n} ({top_scores[n]})" for n in top_names)
            print(f"         top 3: {scores_str}")
            if not hits_found:
                print(f"         MISSING expected: {expected_hits}")
            if misses_found:
                print(f"         UNEXPECTED in top 3: {misses_found}")

        conn.close()

        print(f"\n--- Results: {passed} passed, {failed} failed ---")
        return 0 if failed == 0 else 1


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.exit(main())
