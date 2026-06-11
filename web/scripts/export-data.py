"""Export icon data from SQLite databases to JSON/binary for the web SPA.

Reads from:
  - lucide-icons.db (SVGs)
  - lucide-search.db (descriptions, clusters, embeddings)
  - gemini-icon-descriptions.jsonl (tags, categories)

Writes:
  - public/data/icons.json
  - public/data/umap-coords.json
  - public/data/embeddings-{model}.bin (one per embedding model)

The embedding matrices are the exact vectors fastembed produced at build
time — the browser only embeds the query, so Python and web search score
against identical document vectors.
"""

import importlib.metadata
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import umap

from lucide.config import DEFAULT_SEARCH_MODEL_ID, EMBEDDING_MODELS

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "src" / "lucide" / "data"
ICONS_DB = DATA_DIR / "lucide-icons.db"
SEARCH_DB = DATA_DIR / "lucide-search.db"
DESCRIPTIONS_JSONL = DATA_DIR / "gemini-icon-descriptions.jsonl"
CLUSTERS_JSON = DATA_DIR / "lucide-icon-clusters.json"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "public" / "data"


def _web_models() -> list:
    """Registry entries usable in the browser (server-only models excluded)."""
    return [cfg for cfg in EMBEDDING_MODELS.values() if cfg.web_model]


def _model_manifest() -> list[dict]:
    """Build the web manifest model list from the shared registry."""
    return [
        {
            "id": cfg.id,
            "dim": cfg.dim,
            "file": f"embeddings-{cfg.id}.bin",
            "hfModel": cfg.web_model,
            "pooling": cfg.pooling,
            "dtype": cfg.web_dtype,
            "queryPrefix": cfg.query_prefix,
            "label": cfg.label,
            "default": cfg.id == DEFAULT_SEARCH_MODEL_ID,
            # The model whose embeddings produced the UMAP layout and
            # clusters; Explore must use the same space for neighbor lines
            "clusterSource": cfg.id == DEFAULT_SEARCH_MODEL_ID,
        }
        for cfg in _web_models()
    ]


def _export_names() -> list[str]:
    """Icon names included in the export, in manifest order.

    The embedding matrices are indexed positionally against icons.json, so
    every exporter must use this exact list.
    """
    icons_conn = sqlite3.connect(f"file:{ICONS_DB}?mode=ro", uri=True)
    svgs = {r[0] for r in icons_conn.execute("SELECT name FROM icons")}
    icons_conn.close()

    search_conn = sqlite3.connect(f"file:{SEARCH_DB}?mode=ro", uri=True)
    described = {
        r[0] for r in search_conn.execute("SELECT name FROM icon_descriptions")
    }
    search_conn.close()

    return sorted(svgs & described)


def export_icons(names: list[str]) -> None:
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

    try:
        package_version = importlib.metadata.version("python-lucide")
    except importlib.metadata.PackageNotFoundError:
        package_version = ""

    output = {
        "version": version,
        "packageVersion": package_version,
        "models": _model_manifest(),
        "icons": icons,
    }

    out_path = OUTPUT_DIR / "icons.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")))
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Wrote {len(icons)} icons to {out_path} ({size_mb:.1f} MB)", file=sys.stderr)


def export_embeddings(names: list[str]) -> None:
    """Export per-model embedding matrices as raw float32 binaries.

    Rows follow *names* order; vectors are L2-normalized so the web app can
    score with a plain dot product.
    """
    conn = sqlite3.connect(f"file:{SEARCH_DB}?mode=ro", uri=True)
    for cfg in _web_models():
        model_id = cfg.id
        rows = dict(
            conn.execute(
                "SELECT name, embedding FROM icon_embeddings WHERE model = ?",
                (model_id,),
            )
        )
        missing = [n for n in names if n not in rows]
        if missing:
            conn.close()
            raise SystemExit(
                f"Search DB has no {model_id!r} embeddings for {len(missing)} "
                f"icons (e.g. {missing[:5]}); rebuild with 'make build-search'"
            )

        matrix = np.stack(
            [np.frombuffer(rows[n], dtype=np.float32) for n in names]
        ).astype(np.float32)
        if matrix.shape[1] != cfg.dim:
            conn.close()
            raise SystemExit(
                f"{model_id!r} embeddings are {matrix.shape[1]}d, expected "
                f"{cfg.dim}d; rebuild with 'make build-search'"
            )

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        matrix = (matrix / norms).astype(np.float32)

        out_path = OUTPUT_DIR / f"embeddings-{model_id}.bin"
        out_path.write_bytes(matrix.tobytes())
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(
            f"Wrote {matrix.shape[0]}x{matrix.shape[1]} {model_id} embeddings "
            f"to {out_path} ({size_mb:.1f} MB)",
            file=sys.stderr,
        )
    conn.close()


def _load_cluster_coords() -> tuple[list[str], np.ndarray] | None:
    """Load the UMAP projection persisted by the clustering pipeline."""
    if not CLUSTERS_JSON.exists():
        return None
    raw = json.loads(CLUSTERS_JSON.read_text()).get("coords")
    if not raw:
        return None
    names = sorted(raw)
    return names, np.array([raw[n] for n in names], dtype=np.float64)


def export_umap_coords() -> None:
    """Export the 2D embedding projection as normalized JSON coords.

    Reuses the coords saved by ``lucide cluster`` so the explore map matches
    the cluster assignments exactly; falls back to recomputing UMAP for
    clusters files generated before coords were persisted.
    """
    persisted = _load_cluster_coords()
    if persisted:
        names, coords_array = persisted
        print(
            f"Reusing {len(names)} UMAP coords from {CLUSTERS_JSON.name}",
            file=sys.stderr,
        )
    else:
        conn = sqlite3.connect(f"file:{SEARCH_DB}?mode=ro", uri=True)
        rows = conn.execute(
            "SELECT name, embedding FROM icon_embeddings WHERE model = ? ORDER BY name",
            (DEFAULT_SEARCH_MODEL_ID,),
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
    names = _export_names()
    export_icons(names)
    export_embeddings(names)
    export_umap_coords()


if __name__ == "__main__":
    main()
