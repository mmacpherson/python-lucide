"""Export icon data from SQLite databases to JSON for the web SPA.

Reads from:
  - lucide-icons.db (SVGs)
  - lucide-search.db (descriptions, clusters, embeddings)
  - gemini-icon-descriptions.jsonl (tags, categories)

Writes:
  - public/data/icons.json
  - public/data/umap-coords.json
"""

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import umap

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "src" / "lucide" / "data"
ICONS_DB = DATA_DIR / "lucide-icons.db"
SEARCH_DB = DATA_DIR / "lucide-search.db"
DESCRIPTIONS_JSONL = DATA_DIR / "gemini-icon-descriptions.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "public" / "data"

MODEL_CONFIGS = [
    {
        "id": "minilm",
        "dim": 384,
        "file": "embeddings-minilm.bin",
        "hfModel": "Xenova/all-MiniLM-L6-v2",
        "queryPrefix": "",
        "docPrefix": "",
        "label": "Faster",
    },
    {
        "id": "bge-small",
        "dim": 384,
        "file": "embeddings-bge-small.bin",
        "hfModel": "Xenova/bge-small-en-v1.5",
        # BGE v1.5 asymmetric retrieval: queries need this instruction, documents don't
        "queryPrefix": "Represent this sentence for searching relevant passages: ",
        "docPrefix": "",
        "label": "Better",
    },
]


def export_icons() -> None:
    """Export icon metadata and SVGs to icons.json."""
    icons_conn = sqlite3.connect(f"file:{ICONS_DB}?mode=ro", uri=True)
    search_conn = sqlite3.connect(f"file:{SEARCH_DB}?mode=ro", uri=True)

    version_row = icons_conn.execute(
        "SELECT value FROM metadata WHERE key = 'version'"
    ).fetchone()
    version = version_row[0] if version_row else "unknown"

    svgs = dict(icons_conn.execute("SELECT name, svg FROM icons").fetchall())
    icons_conn.close()

    tags: dict[str, list[str]] = {}
    categories: dict[str, list[str]] = {}
    for line in DESCRIPTIONS_JSONL.read_text().splitlines():
        rec = json.loads(line)
        name = rec["name"]
        if rec.get("tags"):
            tags[name] = rec["tags"]
        if rec.get("categories"):
            categories[name] = rec["categories"]

    descriptions: dict[str, str] = {}
    for name, desc in search_conn.execute(
        "SELECT name, description FROM icon_descriptions"
    ):
        descriptions[name] = desc

    cluster_themes: dict[str, str] = {}
    for name, theme in search_conn.execute("SELECT name, theme FROM icon_clusters"):
        cluster_themes[name] = theme

    search_conn.close()

    names = sorted(n for n in svgs if n in descriptions)

    icons = []
    for name in names:
        icons.append(
            {
                "name": name,
                "svg": svgs[name],
                "description": descriptions[name],
                "tags": tags.get(name, []),
                "categories": categories.get(name, []),
                "cluster": cluster_themes.get(name, ""),
            }
        )

    output = {
        "version": version,
        "models": MODEL_CONFIGS,
        "icons": icons,
    }

    out_path = OUTPUT_DIR / "icons.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")))
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Wrote {len(icons)} icons to {out_path} ({size_mb:.1f} MB)", file=sys.stderr)


def export_umap_coords() -> None:
    """Compute UMAP 2D projection from search DB embeddings and export as JSON."""
    conn = sqlite3.connect(f"file:{SEARCH_DB}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT name, embedding FROM icon_embeddings ORDER BY name"
    ).fetchall()
    conn.close()

    names = [r[0] for r in rows]
    matrix = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    print(f"Loaded {len(names)} embeddings ({matrix.shape[1]}d)", file=sys.stderr)

    print("Running UMAP (2D projection)...", file=sys.stderr)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    coords_array = reducer.fit_transform(matrix)

    # Normalize to 0..1 range
    mins = coords_array.min(axis=0)
    maxs = coords_array.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    normalized = (coords_array - mins) / ranges

    coords = {
        names[i]: [round(float(normalized[i, 0]), 5), round(float(normalized[i, 1]), 5)]
        for i in range(len(names))
    }

    out_path = OUTPUT_DIR / "umap-coords.json"
    out_path.write_text(json.dumps(coords, separators=(",", ":")))
    size_kb = out_path.stat().st_size / 1024
    print(
        f"Wrote {len(coords)} UMAP coords to {out_path} ({size_kb:.1f} KB)",
        file=sys.stderr,
    )


def main() -> None:
    """Export all web app data files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_icons()
    export_umap_coords()


if __name__ == "__main__":
    main()
