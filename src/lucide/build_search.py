"""Build-time pipeline for generating icon search data.

Used by the ``lucide-db`` CLI and CI to produce descriptions (via a VLM) and
embeddings (via fastembed) for semantic icon search.  Not imported at runtime
by library consumers.

Pipeline:
    1. ``generate_descriptions()`` — call VLM, write JSONL
    2. ``build_search_db()`` — read JSONL, compute embeddings, write SQLite

The JSONL file is the durable source of truth for descriptions.  The SQLite
search DB is a derived artifact rebuilt from it.

Requires build-time dependencies: ``cairosvg``, ``fastembed``, and a
``GEMINI_API_KEY`` for VLM description generation.
"""

import base64
import concurrent.futures
import hashlib
import json
import logging
import os
import pathlib
import sqlite3
import subprocess
import threading
import urllib.request
from datetime import datetime, timezone
from typing import TypedDict

from .config import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VLM_MODEL,
    EMBEDDING_DOCUMENT_PREFIX,
)

logger = logging.getLogger(__name__)


class DescriptionRecord(TypedDict):
    """Schema for one line of the descriptions JSONL file."""

    name: str
    description: str
    model: str
    prompt_template_hash: str
    lucide_version: str
    tags: list[str]
    categories: list[str]
    timestamp: str


GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={api_key}"
)

VLM_PROMPT_TEMPLATE = """\
This is a 24\u00d724 SVG icon named "{name}" from the Lucide icon library.
{metadata_line}
Describe this icon in 30\u201360 words. Cover:
1. What it visually depicts
2. Concepts, actions, or emotions it represents

Do not mention where the icon might be used in a UI. \
Return ONLY the description text."""

# Default concurrency for Gemini API calls
_DEFAULT_CONCURRENCY = 20

# Short hash of the prompt template for provenance tracking
_PROMPT_TEMPLATE_HASH = hashlib.sha256(VLM_PROMPT_TEMPLATE.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def load_descriptions_jsonl(
    jsonl_path: pathlib.Path,
) -> dict[str, DescriptionRecord]:
    """Load a descriptions JSONL file into a dict keyed by icon name.

    Args:
        jsonl_path: Path to the JSONL file.

    Returns:
        Dict mapping icon name to the full record.
    """
    records: dict[str, DescriptionRecord] = {}
    with open(jsonl_path) as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            record: DescriptionRecord = json.loads(stripped)
            records[record["name"]] = record
    return records


def _append_jsonl(jsonl_path: pathlib.Path, record: DescriptionRecord) -> None:
    """Append a single JSON record to a JSONL file."""
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Lucide metadata helpers
# ---------------------------------------------------------------------------


def _read_icon_metadata(icons_dir: pathlib.Path, name: str) -> dict[str, list[str]]:
    """Read tags and categories from a Lucide icon's JSON sidecar file."""
    json_path = icons_dir / f"{name}.json"
    if not json_path.exists():
        return {"tags": [], "categories": []}
    try:
        with open(json_path) as f:
            data = json.load(f)
        return {
            "tags": data.get("tags", []),
            "categories": data.get("categories", []),
        }
    except (json.JSONDecodeError, OSError):
        return {"tags": [], "categories": []}


def _build_metadata_line(icons_dir: pathlib.Path | None, name: str) -> str:
    """Build the metadata portion of the VLM prompt."""
    if icons_dir is None:
        return ""
    meta = _read_icon_metadata(icons_dir, name)
    parts = []
    if meta["tags"]:
        parts.append(f"Its existing tags are: {', '.join(meta['tags'])}.")
    if meta["categories"]:
        parts.append(f"Categories: {', '.join(meta['categories'])}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# SVG rendering
# ---------------------------------------------------------------------------


def _render_svg_to_png(svg_content: str, size: int = 96) -> bytes:
    """Render an SVG string to a PNG byte-string.

    Renders at 4\u00d7 native size (96px from 24px source) for better VLM
    interpretation.
    """
    import cairosvg  # noqa: PLC0415

    result: bytes = cairosvg.svg2png(
        bytestring=svg_content.encode("utf-8"),
        output_width=size,
        output_height=size,
    )
    return result


# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------


def _call_gemini(
    prompt: str,
    image_data: bytes,
    api_key: str,
    model: str = DEFAULT_VLM_MODEL,
) -> str | None:
    """Send a prompt + image to the Gemini API and return the text response."""
    url = GEMINI_API_URL.format(model=model, api_key=api_key)
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": base64.b64encode(image_data).decode("ascii"),
                        }
                    },
                ]
            }
        ]
    }
    body = json.dumps(payload).encode("utf-8")
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
        logger.warning("Gemini API call failed for icon", exc_info=True)
    return None


# ---------------------------------------------------------------------------
# Lucide repo management
# ---------------------------------------------------------------------------


def _get_lucide_cache_dir() -> pathlib.Path:
    """Return the cache directory for the Lucide repo clone."""
    return pathlib.Path.home() / ".cache" / "python-lucide" / "lucide-repo"


def ensure_lucide_repo(
    version: str,
    *,
    repo_dir: pathlib.Path | None = None,
) -> pathlib.Path:
    """Ensure a shallow clone of the Lucide repo at the given version.

    Clones to ``~/.cache/python-lucide/lucide-repo/`` if *repo_dir* is not
    provided.  Re-clones if the cached version doesn't match.

    Args:
        version: Lucide version tag (e.g. "0.577.0").
        repo_dir: Explicit path to a Lucide repo checkout.  If provided and
            it contains an ``icons/`` directory, it is used as-is.

    Returns:
        Path to the ``icons/`` directory within the repo.

    Raises:
        RuntimeError: If cloning fails.
    """
    # Use explicit dir if provided and valid
    if repo_dir is not None:
        icons_dir = repo_dir / "icons" if repo_dir.name != "icons" else repo_dir
        if icons_dir.exists():
            return icons_dir
        # If repo_dir is the icons dir itself
        if repo_dir.exists() and any(repo_dir.glob("*.svg")):
            return repo_dir

    cache_dir = _get_lucide_cache_dir()
    version_file = cache_dir / ".lucide-version"

    # Check if cached clone matches the requested version
    if cache_dir.exists() and version_file.exists():
        cached_version = version_file.read_text().strip()
        icons_dir = cache_dir / "icons"
        if cached_version == version and icons_dir.exists():
            logger.info("Using cached Lucide repo (v%s)", version)
            return icons_dir

    # Clone fresh
    import shutil  # noqa: PLC0415

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    repo_url = "https://github.com/lucide-icons/lucide.git"
    logger.info("Cloning Lucide repo v%s to %s", version, cache_dir)

    try:
        subprocess.run(
            [
                "git",
                "-c",
                "advice.detachedHead=false",
                "clone",
                "--quiet",
                "--depth=1",
                f"--branch={version}",
                repo_url,
                str(cache_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to clone Lucide repo at tag {version}: {e.stderr}"
        ) from e
    except FileNotFoundError as e:
        raise RuntimeError(
            "git not found. Install git to generate icon descriptions."
        ) from e

    version_file.write_text(version)

    icons_dir = cache_dir / "icons"
    if not icons_dir.exists():
        raise RuntimeError(f"icons/ directory not found in cloned repo: {cache_dir}")

    return icons_dir


# ---------------------------------------------------------------------------
# Step 1: Generate descriptions → JSONL
# ---------------------------------------------------------------------------


def generate_descriptions(  # noqa: PLR0913, PLR0915
    main_db_path: pathlib.Path,
    jsonl_path: pathlib.Path,
    *,
    icons_dir: pathlib.Path | None = None,
    api_key: str | None = None,
    incremental: bool = True,
    verbose: bool = False,
) -> int:
    """Generate VLM text descriptions and write them to a JSONL file.

    Reads SVG content from *main_db_path*, renders each to PNG, and sends it
    to the Gemini API along with the icon name and Lucide metadata (tags,
    categories) from the repo.

    If *icons_dir* is not provided, the Lucide repo is automatically cloned
    (shallow) to ``~/.cache/python-lucide/lucide-repo/`` at the version
    matching the icons database.

    Args:
        main_db_path: The icons database (read SVGs from here).
        jsonl_path: Output JSONL file (appended to, not overwritten).
        icons_dir: Lucide repo ``icons/`` directory.  Auto-cloned if omitted.
        api_key: Gemini API key.  Falls back to ``GEMINI_API_KEY`` env var.
        incremental: Skip icons that already have entries in the JSONL.
        verbose: Log progress for every icon (not just every 100).

    Returns:
        Number of descriptions generated.
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Gemini API key required. Pass --gemini-api-key or set GEMINI_API_KEY."
        )

    # Read version and icon names from the main DB
    main_conn = sqlite3.connect(f"file:{main_db_path}?mode=ro", uri=True)
    try:
        all_names = [
            row[0]
            for row in main_conn.execute(
                "SELECT name FROM icons ORDER BY name"
            ).fetchall()
        ]
        row = main_conn.execute(
            "SELECT value FROM metadata WHERE key = 'version'"
        ).fetchone()
        lucide_version = row[0] if row else "unknown"
    finally:
        main_conn.close()

    # Ensure we have the Lucide repo for metadata
    icons_dir = ensure_lucide_repo(lucide_version, repo_dir=icons_dir)

    # Determine which icons need descriptions
    existing: set[str] = set()
    if incremental and jsonl_path.exists():
        existing = set(load_descriptions_jsonl(jsonl_path).keys())

    names = [n for n in all_names if n not in existing]
    if not names:
        logger.info("All icons already have descriptions in %s", jsonl_path)
        return 0

    logger.info("Generating descriptions for %d icons", len(names))

    # Read SVG content from the main DB
    main_conn = sqlite3.connect(f"file:{main_db_path}?mode=ro", uri=True)
    svgs: dict[str, str] = {}
    try:
        for name in names:
            row = main_conn.execute(
                "SELECT svg FROM icons WHERE name = ?", (name,)
            ).fetchone()
            if row:
                svgs[name] = row[0]
    finally:
        main_conn.close()

    generated = 0
    write_lock = threading.Lock()

    def _process_icon(name: str) -> bool:
        svg_content = svgs.get(name)
        if not svg_content:
            logger.warning("No SVG found for icon: %s", name)
            return False

        png_data = _render_svg_to_png(svg_content)
        metadata_line = _build_metadata_line(icons_dir, name)
        prompt = VLM_PROMPT_TEMPLATE.format(name=name, metadata_line=metadata_line)
        meta = _read_icon_metadata(icons_dir, name)

        description = _call_gemini(prompt, png_data, api_key)
        if not description:
            logger.warning("No description returned for icon: %s", name)
            return False

        record: DescriptionRecord = {
            "name": name,
            "description": description,
            "model": DEFAULT_VLM_MODEL,
            "prompt_template_hash": _PROMPT_TEMPLATE_HASH,
            "lucide_version": lucide_version,
            "tags": meta["tags"],
            "categories": meta["categories"],
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        with write_lock:
            _append_jsonl(jsonl_path, record)
        return True

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=_DEFAULT_CONCURRENCY
    ) as pool:
        futures = {pool.submit(_process_icon, name): name for name in names}
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            if future.result():
                generated += 1
            if verbose or i % 100 == 0:
                logger.info(
                    "Progress: %d/%d done (%d described)",
                    i,
                    len(names),
                    generated,
                )

    logger.info("Generated %d descriptions", generated)
    return generated


# ---------------------------------------------------------------------------
# Step 2: JSONL → SQLite search DB
# ---------------------------------------------------------------------------


def _ensure_search_tables(conn: sqlite3.Connection) -> None:
    """Create the search DB tables if they don't exist (idempotent)."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS icon_descriptions ("
        "  name TEXT PRIMARY KEY,"
        "  description TEXT NOT NULL,"
        "  model TEXT NOT NULL"
        ")"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS icon_embeddings ("
        "  name TEXT PRIMARY KEY,"
        "  embedding BLOB NOT NULL,"
        "  model TEXT NOT NULL"
        ")"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS metadata ("
        "  key TEXT PRIMARY KEY,"
        "  value TEXT NOT NULL"
        ")"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS icon_clusters ("
        "  name TEXT NOT NULL,"
        "  cluster_id INTEGER NOT NULL,"
        "  theme TEXT NOT NULL"
        ")"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cluster_name ON icon_clusters(name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cluster_theme ON icon_clusters(theme)")
    conn.commit()


def _write_search_db(  # noqa: PLR0913
    search_db_path: pathlib.Path,
    ordered_names: list[str],
    records: dict[str, DescriptionRecord],
    embeddings: list,
    clusters_path: pathlib.Path,
    *,
    version: str | None = None,
    verbose: bool = False,
) -> None:
    """Write descriptions, embeddings, clusters, and metadata to SQLite."""
    import numpy as np  # noqa: PLC0415

    search_db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(search_db_path)
    try:
        _ensure_search_tables(conn)

        conn.execute("DELETE FROM icon_descriptions")
        conn.execute("DELETE FROM icon_embeddings")

        for name in ordered_names:
            rec = records[name]
            conn.execute(
                "INSERT INTO icon_descriptions (name, description, model)"
                " VALUES (?, ?, ?)",
                (name, rec["description"], rec["model"]),
            )

        for name, emb in zip(ordered_names, embeddings, strict=True):
            blob = np.array(emb, dtype=np.float32).tobytes()
            conn.execute(
                "INSERT INTO icon_embeddings (name, embedding, model) VALUES (?, ?, ?)",
                (name, blob, DEFAULT_EMBEDDING_MODEL),
            )

        # Load clusters
        conn.execute("DELETE FROM icon_clusters")
        cluster_data = json.loads(clusters_path.read_text())
        for cid, cluster in cluster_data["clusters"].items():
            theme = cluster.get("theme", f"Cluster {cid}")
            if cid == "-1":
                continue
            for icon_name in cluster["icons"]:
                conn.execute(
                    "INSERT INTO icon_clusters (name, cluster_id, theme)"
                    " VALUES (?, ?, ?)",
                    (icon_name, int(cid), theme),
                )

        # Metadata
        first = records[ordered_names[0]]
        resolved_version = version or first.get("lucide_version", "unknown")
        now = datetime.now(tz=timezone.utc).isoformat()
        for key, value in [
            ("version", resolved_version),
            ("embedding_model", DEFAULT_EMBEDDING_MODEL),
            ("embedding_dim", str(DEFAULT_EMBEDDING_DIM)),
            ("description_model", DEFAULT_VLM_MODEL),
            ("built_at", now),
        ]:
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                (key, value),
            )

        conn.commit()
        conn.execute("VACUUM")
    finally:
        conn.close()

    if verbose:
        size_kb = search_db_path.stat().st_size / 1024
        logger.info("Search DB written: %s (%.0f KB)", search_db_path, size_kb)


def _read_icons_db_info(
    icons_db_path: pathlib.Path,
) -> tuple[set[str], str | None]:
    """Read icon names and version from the icons database."""
    conn = sqlite3.connect(icons_db_path)
    try:
        names = {r[0] for r in conn.execute("SELECT name FROM icons").fetchall()}
        row = conn.execute(
            "SELECT value FROM metadata WHERE key = 'version'"
        ).fetchone()
        return names, row[0] if row else None
    finally:
        conn.close()


def build_search_db(
    jsonl_path: pathlib.Path,
    search_db_path: pathlib.Path,
    clusters_path: pathlib.Path,
    *,
    icons_db_path: pathlib.Path | None = None,
    verbose: bool = False,
) -> None:
    """Build the SQLite search database from descriptions, embeddings, and clusters.

    Reads descriptions from *jsonl_path*, computes embeddings with fastembed,
    loads cluster assignments from *clusters_path*, and writes everything to
    *search_db_path*.  The DB is rebuilt from scratch each time.

    When *icons_db_path* is provided, only icons present in that database are
    included and the version metadata is read from it.

    Args:
        jsonl_path: Input JSONL file with VLM descriptions.
        search_db_path: Output SQLite database path.
        clusters_path: JSON file with cluster assignments.
        icons_db_path: Optional icons database for filtering and version.
        verbose: Verbose logging.
    """
    from fastembed import TextEmbedding  # noqa: PLC0415

    records = load_descriptions_jsonl(jsonl_path)
    if not records:
        logger.warning("No descriptions found in %s", jsonl_path)
        return

    # Filter to only icons present in the icons DB when provided
    icons_version: str | None = None
    if icons_db_path is not None:
        valid_names, icons_version = _read_icons_db_info(icons_db_path)
        skipped = sorted(set(records.keys()) - valid_names)
        if skipped:
            logger.info("Skipping %d stale descriptions: %s", len(skipped), skipped)
        ordered_names = sorted(n for n in records if n in valid_names)
    else:
        ordered_names = sorted(records.keys())

    logger.info("Building search DB from %d descriptions", len(ordered_names))

    # Build embedding input: name + tags + categories + description
    def _embedding_text(rec: DescriptionRecord) -> str:
        parts = [rec["name"].replace("-", " ")]
        if rec["tags"]:
            parts.append(f"Tags: {', '.join(rec['tags'])}")
        if rec["categories"]:
            parts.append(f"Categories: {', '.join(rec['categories'])}")
        parts.append(rec["description"])
        return f"{EMBEDDING_DOCUMENT_PREFIX}{'. '.join(parts)}"

    texts = [_embedding_text(records[n]) for n in ordered_names]
    embedder = TextEmbedding(model_name=DEFAULT_EMBEDDING_MODEL)
    embeddings = list(embedder.embed(texts))
    logger.info("Computed %d embeddings", len(embeddings))

    _write_search_db(
        search_db_path,
        ordered_names,
        records,
        embeddings,
        clusters_path,
        version=icons_version,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------


def build_search_data(  # noqa: PLR0913
    main_db_path: pathlib.Path,
    search_db_path: pathlib.Path,
    clusters_path: pathlib.Path,
    *,
    jsonl_path: pathlib.Path | None = None,
    icons_dir: pathlib.Path | None = None,
    api_key: str | None = None,
    incremental: bool = True,
    verbose: bool = False,
) -> None:
    """Run the full search data pipeline: descriptions then build DB.

    Args:
        main_db_path: The icons database (source of SVGs).
        search_db_path: The search database (output SQLite).
        clusters_path: JSON file with cluster assignments.
        jsonl_path: Path for the descriptions JSONL.  Defaults to
            ``gemini-icon-descriptions.jsonl`` alongside the search DB.
        icons_dir: Optional Lucide repo ``icons/`` directory for metadata.
        api_key: Gemini API key.
        incremental: Only generate descriptions for new icons.
        verbose: Verbose logging.
    """
    if jsonl_path is None:
        jsonl_path = search_db_path.parent / "gemini-icon-descriptions.jsonl"

    generate_descriptions(
        main_db_path,
        jsonl_path,
        icons_dir=icons_dir,
        api_key=api_key,
        incremental=incremental,
        verbose=verbose,
    )

    build_search_db(
        jsonl_path,
        search_db_path,
        clusters_path,
        icons_db_path=main_db_path,
        verbose=verbose,
    )

    logger.info("Search data complete: %s", search_db_path)


# Keep this around for the test suite and backward compat with the CLI
def ensure_search_tables(search_db_path: pathlib.Path) -> None:
    """Create the search DB tables if they don't exist (idempotent).

    Args:
        search_db_path: Path to the search SQLite database file.
    """
    conn = sqlite3.connect(search_db_path)
    try:
        _ensure_search_tables(conn)
    finally:
        conn.close()
