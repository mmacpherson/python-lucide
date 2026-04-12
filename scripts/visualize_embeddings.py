#!/usr/bin/env python3
"""Generate an interactive HTML visualization of the Lucide icon embeddings.

Uses UMAP to project 768d embeddings to 2D, colors by Lucide category,
and builds an interactive Plotly scatter plot with hover details.

Usage:
    uv run python scripts/visualize_embeddings.py
"""

import json
import pathlib
import sqlite3

import numpy as np
import plotly.graph_objects as go
import umap

# --- Config ---

SEARCH_DB = pathlib.Path("src/lucide/data/lucide-search.db")
JSONL = pathlib.Path("src/lucide/data/gemini-icon-descriptions.jsonl")
OUTPUT = pathlib.Path("lucide-embedding-map.html")

# Category color palette — hand-picked for visual distinction
CATEGORY_COLORS = {
    "arrows": "#e6194b",
    "text": "#3cb44b",
    "shapes": "#ffe119",
    "layout": "#4363d8",
    "files": "#f58231",
    "communication": "#911eb4",
    "devices": "#42d4f4",
    "photography": "#f032e6",
    "security": "#bfef45",
    "account": "#fabed4",
    "social": "#469990",
    "finance": "#dcbeff",
    "weather": "#9a6324",
    "transportation": "#fffac8",
    "medical": "#800000",
    "tools": "#aaffc3",
    "gaming": "#808000",
    "multimedia": "#ffd8b1",
    "nature": "#228b22",
    "animals": "#8b4513",
    "food-beverage": "#ff6347",
    "accessibility": "#00ced1",
    "home": "#dda0dd",
    "development": "#708090",
    "math": "#b0c4de",
    "science": "#20b2aa",
    "time": "#cd853f",
    "navigation": "#6495ed",
    "sustainability": "#2e8b57",
    "mail": "#ba55d3",
    "notifications": "#ff4500",
    "connectivity": "#1e90ff",
    "brands": "#ff69b4",
    "cursors": "#a0522d",
    "charts": "#5f9ea0",
    "design": "#db7093",
    "editing": "#4682b4",
    "emoji": "#ffa07a",
    "furniture": "#8fbc8f",
    "seasons": "#d2691e",
    "maps": "#556b2f",
    "money": "#b8860b",
    "shopping": "#9370db",
}
FALLBACK_COLOR = "#999999"


def main() -> None:
    # Load embeddings from search DB
    conn = sqlite3.connect(f"file:{SEARCH_DB}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT e.name, e.embedding, d.description "
        "FROM icon_embeddings e "
        "JOIN icon_descriptions d ON e.name = d.name "
        "ORDER BY e.name"
    ).fetchall()
    conn.close()

    names = [r[0] for r in rows]
    descriptions = [r[2] for r in rows]
    matrix = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])

    print(f"Loaded {len(names)} embeddings ({matrix.shape[1]}d)")

    # Load categories from JSONL
    jsonl_data = {}
    if JSONL.exists():
        for line in JSONL.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                jsonl_data[rec["name"]] = rec

    # Assign primary category (first category, or "uncategorized")
    categories = []
    all_categories = []
    for name in names:
        rec = jsonl_data.get(name, {})
        cats = rec.get("categories", [])
        primary = cats[0] if cats else "uncategorized"
        categories.append(primary)
        all_categories.append(", ".join(cats) if cats else "uncategorized")

    # UMAP projection
    print("Running UMAP (2D projection)...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(matrix)
    print("UMAP complete")

    # Get unique categories sorted by frequency
    cat_counts: dict[str, int] = {}
    for cat in categories:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    sorted_cats = sorted(cat_counts.keys(), key=lambda c: -cat_counts[c])

    # Build one trace per category for a nice legend
    fig = go.Figure()

    for cat in sorted_cats:
        mask = [i for i, c in enumerate(categories) if c == cat]
        color = CATEGORY_COLORS.get(cat, FALLBACK_COLOR)

        hover_texts = [
            f"<b>{names[i]}</b><br>"
            f"<i>{all_categories[i]}</i><br>"
            f"<br>{descriptions[i][:120]}..."
            if len(descriptions[i]) > 120
            else f"<b>{names[i]}</b><br>"
            f"<i>{all_categories[i]}</i><br>"
            f"<br>{descriptions[i]}"
            for i in mask
        ]

        fig.add_trace(
            go.Scatter(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                name=f"{cat} ({cat_counts[cat]})",
                marker=dict(
                    size=5,
                    color=color,
                    opacity=0.7,
                    line=dict(width=0.5, color="white"),
                ),
                text=hover_texts,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=dict(
            text=(
                "Lucide Icon Embeddings — "
                "UMAP projection of 1,703 icons (nomic-embed-text-v1.5-Q, 768d)"
            ),
            font=dict(size=16),
        ),
        width=1400,
        height=900,
        template="plotly_dark",
        legend=dict(
            title="Primary Category",
            font=dict(size=10),
            itemsizing="constant",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
        xaxis=dict(showgrid=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, showticklabels=False, title=""),
        hovermode="closest",
    )

    fig.write_html(
        str(OUTPUT),
        include_plotlyjs=True,
        full_html=True,
    )
    size_mb = OUTPUT.stat().st_size / (1024 * 1024)
    print(f"Written: {OUTPUT} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
