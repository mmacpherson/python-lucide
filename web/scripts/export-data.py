"""Export icon data from SQLite databases to JSON for the web SPA.

Reads from:
  - lucide-icons.db (SVGs)
  - lucide-search.db (descriptions, clusters)
  - gemini-icon-descriptions.jsonl (tags, categories)

Writes:
  - public/data/icons.json
"""

import json
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "src" / "lucide" / "data"
ICONS_DB = DATA_DIR / "lucide-icons.db"
SEARCH_DB = DATA_DIR / "lucide-search.db"
DESCRIPTIONS_JSONL = DATA_DIR / "gemini-icon-descriptions.jsonl"
OUTPUT = Path(__file__).resolve().parent.parent / "public" / "data" / "icons.json"

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
        "queryPrefix": "",
        "docPrefix": "",
        "label": "Better",
    },
]


def main() -> None:
    """Export icon data from SQLite DBs and JSONL to icons.json for the web app."""
    icons_conn = sqlite3.connect(f"file:{ICONS_DB}?mode=ro", uri=True)
    search_conn = sqlite3.connect(f"file:{SEARCH_DB}?mode=ro", uri=True)

    version_row = icons_conn.execute(
        "SELECT value FROM metadata WHERE key = 'version'"
    ).fetchone()
    version = version_row[0] if version_row else "unknown"

    svgs = dict(icons_conn.execute("SELECT name, svg FROM icons").fetchall())
    icons_conn.close()

    # Tags and categories come from the JSONL, not the icons DB
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

    # Only include icons that have descriptions (i.e., are in the search DB)
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

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")))
    size_mb = OUTPUT.stat().st_size / 1024 / 1024
    print(f"Wrote {len(icons)} icons to {OUTPUT} ({size_mb:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
