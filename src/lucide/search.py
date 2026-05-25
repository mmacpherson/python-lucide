"""Semantic search for Lucide icons.

Provides embedding-based search so users can find icons by natural language
queries like "payment", "hard work", or "bird". Search data is downloaded
on first use and cached locally.

Requires the ``search`` extra: ``pip install 'python-lucide[search]'``
"""

import logging
import os
import pathlib
import sqlite3
import urllib.request
from typing import Any, NamedTuple

from .config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LUCIDE_TAG,
    EMBEDDING_QUERY_PREFIX,
    SEARCH_DB_URL_TEMPLATE,
)
from .db import get_db_connection

logger = logging.getLogger(__name__)


class SearchNotAvailableError(Exception):
    """Raised when search is attempted but search data is not available."""


class SearchResult(NamedTuple):
    """A single icon search result."""

    name: str
    score: float
    description: str


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------


def _check_search_deps() -> None:
    """Raise a clear error if the search extra is not installed."""
    try:
        import fastembed  # noqa: F401, PLC0415
    except ImportError:
        raise ImportError(
            "Semantic search requires the 'search' extra. "
            "Install with: pip install 'python-lucide[search]'"
        ) from None


# ---------------------------------------------------------------------------
# Search DB resolution and download
# ---------------------------------------------------------------------------


def _get_icons_version() -> str:
    """Read the Lucide icon-set version from the main database metadata."""
    with get_db_connection() as conn:
        if conn is None:
            return DEFAULT_LUCIDE_TAG
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM metadata WHERE key = 'version'")
            row = cursor.fetchone()
            return row[0] if row else DEFAULT_LUCIDE_TAG
        except sqlite3.Error:
            return DEFAULT_LUCIDE_TAG


def _get_cache_dir() -> pathlib.Path:
    """Return (and create) the local cache directory for search data."""
    cache_dir = pathlib.Path.home() / ".cache" / "python-lucide"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _resolve_search_db(*, allow_download: bool = False) -> pathlib.Path | None:
    """Locate the search database file.

    Resolution order:
        1. ``LUCIDE_SEARCH_DB_PATH`` environment variable
        2. Cached file in ``~/.cache/python-lucide/``
        3. Download from GitHub release (only when *allow_download* is True)

    Args:
        allow_download: If True, download the search DB when no local copy
            exists.

    Returns:
        Path to the search database, or None if unavailable.

    Raises:
        SearchNotAvailableError: If *allow_download* is True but the download
            fails.
    """
    env_path = os.environ.get("LUCIDE_SEARCH_DB_PATH")
    if env_path:
        path = pathlib.Path(env_path)
        if path.exists():
            return path

    version = _get_icons_version()
    cached = _get_cache_dir() / f"lucide-search-{version}.db"
    if cached.exists():
        return cached

    if allow_download:
        url = SEARCH_DB_URL_TEMPLATE.format(version=version)
        logger.info("Downloading search database from %s", url)
        try:
            urllib.request.urlretrieve(url, cached)
            return cached
        except Exception as e:
            if cached.exists():
                cached.unlink()
            raise SearchNotAvailableError(
                f"Failed to download search database: {e}\n"
                f"URL: {url}\n"
                "Build search data locally with: lucide-db --search-only"
            ) from e

    return None


# ---------------------------------------------------------------------------
# In-memory search index (module-level cache)
# ---------------------------------------------------------------------------

_embedder_instance: Any = None
_search_index: dict[str, Any] | None = None


def _get_embedder() -> Any:
    """Return a cached fastembed TextEmbedding instance."""
    global _embedder_instance  # noqa: PLW0603
    if _embedder_instance is None:
        _check_search_deps()
        import warnings  # noqa: PLC0415

        from fastembed import TextEmbedding  # noqa: PLC0415

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _embedder_instance = TextEmbedding(model_name=DEFAULT_EMBEDDING_MODEL)
    return _embedder_instance


def _load_search_index() -> dict[str, Any]:
    """Load and cache the full search index from the search database.

    The index contains pre-normalized embedding vectors for fast cosine
    similarity via a single matrix-vector dot product.
    """
    global _search_index  # noqa: PLW0603
    import numpy as np  # noqa: PLC0415

    version = _get_icons_version()
    if _search_index is not None and _search_index["version"] == version:
        return _search_index

    db_path = _resolve_search_db(allow_download=True)
    if db_path is None:
        raise SearchNotAvailableError(
            "Search database not found. "
            "Build search data locally with: lucide-db --search-only"
        )

    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT e.name, e.embedding, d.description "
            "FROM icon_embeddings e "
            "JOIN icon_descriptions d ON e.name = d.name "
            "ORDER BY e.name"
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    if not rows:
        raise SearchNotAvailableError("Search database contains no embeddings")

    names = [row[0] for row in rows]
    descriptions = {row[0]: row[2] for row in rows}
    matrix = np.stack([np.frombuffer(row[1], dtype=np.float32) for row in rows])

    # Pre-normalize so cosine similarity is a plain dot product
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    matrix = (matrix / norms).astype(np.float32)

    _search_index = {
        "version": version,
        "names": names,
        "matrix": matrix,
        "descriptions": descriptions,
    }
    return _search_index


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search_icons(
    query: str,
    limit: int = 10,
    threshold: float = 0.0,
) -> list[SearchResult]:
    """Search for icons by natural language query.

    On the first call the search database is downloaded (~8 MB) and cached
    in ``~/.cache/python-lucide/``.  The embedding model (~45 MB) is also
    downloaded once by *fastembed*.

    Args:
        query: Natural language search query (e.g. "payment", "hard work").
        limit: Maximum number of results to return.
        threshold: Minimum cosine-similarity score (0.0-1.0) to include.

    Returns:
        Results ordered by descending similarity score.

    Raises:
        ImportError: If the ``search`` extra is not installed.
        SearchNotAvailableError: If search data cannot be loaded.
    """
    _check_search_deps()
    import numpy as np  # noqa: PLC0415

    embedder = _get_embedder()
    index = _load_search_index()

    prefixed_query = f"{EMBEDDING_QUERY_PREFIX}{query}"
    query_vec = np.array(next(iter(embedder.embed([prefixed_query]))), dtype=np.float32)
    norm = np.linalg.norm(query_vec)
    if norm > 0:
        query_vec /= norm

    scores = index["matrix"] @ query_vec

    results = [
        SearchResult(
            name=index["names"][i],
            score=float(scores[i]),
            description=index["descriptions"][index["names"][i]],
        )
        for i in range(len(index["names"]))
        if scores[i] >= threshold
    ]
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]


def get_icon_description(icon_name: str) -> str | None:
    """Get the VLM-generated text description for an icon.

    Does not trigger a download — returns None if the search database is
    not cached locally.

    Args:
        icon_name: The icon name (e.g. "heart", "credit-card").

    Returns:
        The description string, or None if not available.
    """
    db_path = _resolve_search_db(allow_download=False)
    if db_path is None:
        return None

    try:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT description FROM icon_descriptions WHERE name = ?",
                (icon_name,),
            )
            row = cursor.fetchone()
            return row[0] if row else None
        finally:
            conn.close()
    except sqlite3.Error:
        return None


def search_available() -> bool:
    """Check whether search data is cached locally (without downloading).

    Returns:
        True if the search database exists and contains embeddings.
    """
    db_path = _resolve_search_db(allow_download=False)
    if db_path is None:
        return False

    try:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM icon_embeddings")
            count: int = cursor.fetchone()[0]
            return count > 0
        finally:
            conn.close()
    except sqlite3.Error:
        return False
