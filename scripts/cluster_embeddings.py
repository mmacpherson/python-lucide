#!/usr/bin/env python3
"""Discover clusters in the Lucide icon embedding space.

1. Load embeddings from the search DB
2. HDBSCAN clustering in embedding space
3. Output cluster assignments as JSON for naming by Claude Code subagents
4. Build interactive Plotly visualization

Usage:
    uv run python scripts/cluster_embeddings.py
    # Then run name_clusters.py to add LLM-generated theme names
"""

import json
import pathlib
import sqlite3

import hdbscan
import numpy as np
import plotly.graph_objects as go
import umap

SEARCH_DB = pathlib.Path("src/lucide/data/lucide-search.db")
CLUSTERS_JSON = pathlib.Path("scripts/clusters.json")
OUTPUT = pathlib.Path("lucide-cluster-map.html")
MIN_CLUSTER_SIZE = 5


def discover_clusters() -> dict:
    """Load embeddings, cluster, return structure for naming."""
    conn = sqlite3.connect(f"file:{SEARCH_DB}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT e.name, e.embedding, d.description "
        "FROM icon_embeddings e "
        "JOIN icon_descriptions d ON e.name = d.name "
        "ORDER BY e.name"
    ).fetchall()
    conn.close()

    names = [r[0] for r in rows]
    descriptions = {r[0]: r[2] for r in rows}
    matrix = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    print(f"Loaded {len(names)} embeddings")

    # UMAP
    print("Running UMAP...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(matrix)

    # HDBSCAN on UMAP coordinates (clusters better in low-d)
    print("Running HDBSCAN on UMAP projection...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(coords)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"Found {n_clusters} clusters, {n_noise} noise points")

    # Build cluster data
    clusters: dict[str, dict] = {}
    for i, label in enumerate(labels):
        lid = str(label)
        if lid not in clusters:
            clusters[lid] = {"icons": [], "theme": None}
        clusters[lid]["icons"].append(names[i])

    # Sort by size
    for lid in clusters:
        clusters[lid]["icons"].sort()

    # Save for naming
    data = {
        "clusters": clusters,
        "coords": {
            names[i]: [float(coords[i, 0]), float(coords[i, 1])]
            for i in range(len(names))
        },
        "descriptions": descriptions,
    }

    CLUSTERS_JSON.write_text(json.dumps(data, indent=2))
    print(f"Saved cluster data to {CLUSTERS_JSON}")
    print("\nCluster sizes:")
    for lid in sorted(clusters, key=lambda l: -len(clusters[l]["icons"])):
        icons = clusters[lid]["icons"]
        label = "noise" if lid == "-1" else f"cluster {lid}"
        sample = ", ".join(icons[:8])
        print(f"  {label} ({len(icons)}): {sample}...")

    return data


def build_visualization(data: dict) -> None:
    """Build the HTML visualization from cluster data (with or without names)."""
    import colorsys

    clusters = data["clusters"]
    coords = data["coords"]
    descriptions = data["descriptions"]

    # Generate colors
    real_clusters = [l for l in clusters if l != "-1"]
    n = len(real_clusters)

    def _colors(n: int) -> list[str]:
        out = []
        for i in range(n):
            h = i / n
            r, g, b = colorsys.hsv_to_rgb(h, 0.7, 0.9)
            out.append(f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})")
        return out

    palette = _colors(n)
    label_colors = {l: palette[i] for i, l in enumerate(real_clusters)}
    label_colors["-1"] = "rgb(80,80,80)"

    fig = go.Figure()

    # Sort: largest clusters first, noise last
    sorted_labels = sorted(real_clusters, key=lambda l: -len(clusters[l]["icons"]))
    if "-1" in clusters:
        sorted_labels.append("-1")

    for lid in sorted_labels:
        icons = clusters[lid]["icons"]
        theme = clusters[lid].get("theme") or (
            f"cluster {lid}" if lid != "-1" else "unclustered"
        )
        color = label_colors[lid]
        is_noise = lid == "-1"

        hover = [
            f"<b>{name}</b><br><i>{theme}</i><br><br>{descriptions.get(name, '')[:150]}"
            for name in icons
        ]

        xs = [coords[name][0] for name in icons]
        ys = [coords[name][1] for name in icons]

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=f"{theme} ({len(icons)})",
                marker=dict(
                    size=3 if is_noise else 5,
                    color=color,
                    opacity=0.3 if is_noise else 0.8,
                    line=dict(width=0.5, color="white"),
                ),
                text=hover,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=dict(
            text=(
                "Lucide Icon Embedding Clusters \u2014 themes discovered via HDBSCAN"
            ),
            font=dict(size=16),
        ),
        width=1400,
        height=900,
        template="plotly_dark",
        legend=dict(
            title="Discovered Themes",
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

    fig.write_html(str(OUTPUT), include_plotlyjs=True, full_html=True)
    size_mb = OUTPUT.stat().st_size / (1024 * 1024)
    print(f"Written: {OUTPUT} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    data = discover_clusters()
    build_visualization(data)
