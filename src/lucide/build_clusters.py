"""Discover semantic clusters in the icon embedding space and name them.

Uses UMAP for dimensionality reduction, HDBSCAN for density-based
clustering, and Gemini Flash to generate evocative theme names from
the icon names in each cluster.

Build-time only — requires ``umap-learn``, ``hdbscan``, and ``plotly``
(all in the dev dependency group).
"""

import json
import logging
import os
import pathlib
import sqlite3
import urllib.request
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

CLUSTER_NAMING_MODEL = "gemini-2.5-flash"

NAMING_PROMPT_TEMPLATE = """\
Here are icon names that form a visual/semantic cluster: {icon_names}

Give this cluster a short, evocative theme name (2-4 words).
Be specific about what unifies these icons.
Do not use generic labels like "UI elements" or "miscellaneous".

Return ONLY the theme name, nothing else."""

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={api_key}"
)


def _call_gemini_text(prompt: str, api_key: str) -> str | None:
    """Call Gemini API with a text-only prompt."""
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    body = json.dumps(payload).encode("utf-8")
    url = GEMINI_API_URL.format(model=CLUSTER_NAMING_MODEL, api_key=api_key)
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        candidates = result.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                text: str = parts[0].get("text", "")
                return text.strip()
    except Exception:
        logger.warning("Gemini API call failed", exc_info=True)
    return None


def discover_clusters(
    search_db_path: pathlib.Path,
    *,
    min_cluster_size: int = 5,
) -> dict:
    """Run UMAP + HDBSCAN on the icon embeddings.

    Args:
        search_db_path: Path to the search SQLite database.
        min_cluster_size: Minimum icons to form a cluster.

    Returns:
        Dict with keys: clusters, coords, descriptions.
    """
    import numpy as np  # noqa: PLC0415
    import umap  # noqa: PLC0415
    from hdbscan import HDBSCAN  # noqa: PLC0415

    conn = sqlite3.connect(f"file:{search_db_path}?mode=ro", uri=True)
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
    logger.info("Loaded %d embeddings (%dd)", len(names), matrix.shape[1])

    logger.info("Running UMAP (2D projection)...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    coords_array = reducer.fit_transform(matrix)

    logger.info("Running HDBSCAN (min_cluster_size=%d)...", min_cluster_size)
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(coords_array)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    logger.info("Found %d clusters, %d unclustered points", n_clusters, n_noise)

    clusters: dict[str, dict] = {}
    for i, label in enumerate(labels):
        lid = str(int(label))
        if lid not in clusters:
            clusters[lid] = {"icons": [], "theme": None}
        clusters[lid]["icons"].append(names[i])

    for lid in clusters:
        clusters[lid]["icons"].sort()

    coords = {
        names[i]: [float(coords_array[i, 0]), float(coords_array[i, 1])]
        for i in range(len(names))
    }

    return {
        "clusters": clusters,
        "coords": coords,
        "descriptions": descriptions,
    }


def name_clusters(
    data: dict,
    *,
    api_key: str | None = None,
) -> dict:
    """Name each cluster using Gemini Flash.

    Args:
        data: Output from ``discover_clusters()``.
        api_key: Gemini API key. Falls back to ``GEMINI_API_KEY`` env var.

    Returns:
        The same data dict with ``theme`` populated for each cluster.
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Gemini API key required. Set GEMINI_API_KEY.")

    clusters = data["clusters"]
    for lid in sorted(clusters, key=lambda k: -len(clusters[k]["icons"])):
        if lid == "-1":
            clusters[lid]["theme"] = "Unclustered"
            continue

        icons = clusters[lid]["icons"]
        # Cap at 40 names to keep prompt short
        icon_names = ", ".join(icons[:40])
        prompt = NAMING_PROMPT_TEMPLATE.format(icon_names=icon_names)
        theme = _call_gemini_text(prompt, api_key)

        if theme:
            # Strip quotes if the model wraps it
            theme = theme.strip("\"'")
            clusters[lid]["theme"] = theme
            logger.info("Cluster %s (%d icons): %s", lid, len(icons), theme)
        else:
            clusters[lid]["theme"] = f"Cluster {lid}"
            logger.warning("Failed to name cluster %s", lid)

    return data


def save_clusters_json(
    data: dict,
    output_path: pathlib.Path,
) -> None:
    """Save cluster data to JSON."""
    # Strip descriptions from the output to keep it lean
    out = {
        "clusters": data["clusters"],
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "naming_model": CLUSTER_NAMING_MODEL,
    }
    output_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info("Saved clusters to %s", output_path)


def build_cluster_visualization(
    data: dict,
    output_path: pathlib.Path,
) -> None:
    """Build an interactive HTML scatter plot of the clusters."""
    import colorsys  # noqa: PLC0415

    import plotly.graph_objects as go  # noqa: PLC0415

    clusters = data["clusters"]
    coords = data["coords"]
    descriptions = data["descriptions"]

    real_clusters = [cid for cid in clusters if cid != "-1"]
    n = len(real_clusters)

    palette = []
    for i in range(n):
        h = i / n
        r, g, b = colorsys.hsv_to_rgb(h, 0.7, 0.9)
        palette.append(f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})")

    label_colors = {cid: palette[i] for i, cid in enumerate(real_clusters)}
    label_colors["-1"] = "rgb(80,80,80)"

    fig = go.Figure()

    sorted_labels = sorted(real_clusters, key=lambda k: -len(clusters[k]["icons"]))
    if "-1" in clusters:
        sorted_labels.append("-1")

    for lid in sorted_labels:
        icons = clusters[lid]["icons"]
        theme = clusters[lid].get("theme") or f"Cluster {lid}"
        color = label_colors[lid]
        is_noise = lid == "-1"

        hover = [
            f"<b>{name}</b><br><i>{theme}</i><br><br>{descriptions.get(name, '')[:150]}"
            for name in icons
        ]

        fig.add_trace(
            go.Scatter(
                x=[coords[n][0] for n in icons],
                y=[coords[n][1] for n in icons],
                mode="markers",
                name=f"{theme} ({len(icons)})",
                marker={
                    "size": 3 if is_noise else 5,
                    "color": color,
                    "opacity": 0.3 if is_noise else 0.8,
                    "line": {"width": 0.5, "color": "white"},
                },
                text=hover,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title={
            "text": (
                "Lucide Icon Embedding Clusters \u2014 "
                "themes discovered via HDBSCAN in 768d space"
            ),
            "font": {"size": 16},
        },
        width=1400,
        height=900,
        template="plotly_dark",
        legend={
            "title": "Discovered Themes",
            "font": {"size": 10},
            "itemsizing": "constant",
            "yanchor": "top",
            "y": 1,
            "xanchor": "left",
            "x": 1.02,
        },
        xaxis={"showgrid": False, "showticklabels": False, "title": ""},
        yaxis={"showgrid": False, "showticklabels": False, "title": ""},
        hovermode="closest",
    )

    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Visualization written: %s (%.1f MB)", output_path, size_mb)
