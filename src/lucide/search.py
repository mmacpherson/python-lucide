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
    DEFAULT_LUCIDE_TAG,
    DEFAULT_SEARCH_MODEL_ID,
    EMBEDDING_MODELS,
    SEARCH_DB_SCHEMA_VERSION,
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


def _read_schema_version(path: pathlib.Path) -> int:
    """Read the schema version a search DB declares about itself.

    Returns 1 for pre-versioning databases (the key was introduced with
    schema 2) and 0 for files that cannot be read as a search DB at all.
    """
    try:
        uri = f"file:{path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM metadata WHERE key = 'schema_version'")
            row = cursor.fetchone()
            return int(row[0]) if row else 1
        finally:
            conn.close()
    except (sqlite3.Error, ValueError):
        return 0


def _resolve_search_db(*, allow_download: bool = False) -> pathlib.Path | None:
    """Locate the search database file.

    Resolution order:
        1. ``LUCIDE_SEARCH_DB_PATH`` environment variable
        2. Cached file in ``~/.cache/python-lucide/``
        3. Download from GitHub release (only when *allow_download* is True)

    A cached file whose self-declared schema version doesn't match this
    package's ``SEARCH_DB_SCHEMA_VERSION`` (e.g. left behind by a different
    package version, or corrupted) is deleted and re-downloaded.

    Args:
        allow_download: If True, download the search DB when no local copy
            exists.

    Returns:
        Path to the search database, or None if unavailable.

    Raises:
        SearchNotAvailableError: If *allow_download* is True but the download
            fails or yields a DB with an incompatible schema.
    """
    env_path = os.environ.get("LUCIDE_SEARCH_DB_PATH")
    if env_path:
        path = pathlib.Path(env_path)
        if path.exists():
            return path

    version = _get_icons_version()
    cached = _get_cache_dir() / f"lucide-search-{version}.db"
    if cached.exists():
        if _read_schema_version(cached) == SEARCH_DB_SCHEMA_VERSION:
            return cached
        cached.unlink()

    if allow_download:
        url = SEARCH_DB_URL_TEMPLATE.format(version=version)
        logger.info("Downloading search database from %s", url)
        try:
            urllib.request.urlretrieve(url, cached)
        except Exception as e:
            if cached.exists():
                cached.unlink()
            raise SearchNotAvailableError(
                f"Failed to download search database: {e}\n"
                f"URL: {url}\n"
                "Build search data locally with: lucide build-search"
            ) from e
        downloaded_schema = _read_schema_version(cached)
        if downloaded_schema != SEARCH_DB_SCHEMA_VERSION:
            cached.unlink()
            raise SearchNotAvailableError(
                f"Downloaded search database has schema {downloaded_schema}, "
                f"but this version of python-lucide requires "
                f"{SEARCH_DB_SCHEMA_VERSION}. The published search data may "
                "not match this package version yet.\n"
                "Build search data locally with: lucide build-search"
            )
        return cached

    return None


# ---------------------------------------------------------------------------
# In-memory search index (module-level cache)
# ---------------------------------------------------------------------------

_embedder_instances: dict[str, Any] = {}
_search_indexes: dict[str, dict[str, Any]] = {}


def _resolve_model(model: str) -> str:
    """Validate a model id against the registry."""
    if model not in EMBEDDING_MODELS:
        raise ValueError(
            f"Unknown embedding model {model!r}. "
            f"Available: {', '.join(sorted(EMBEDDING_MODELS))}"
        )
    return model


def _get_embedder(model: str) -> Any:
    """Return a cached fastembed TextEmbedding instance for *model*."""
    if model not in _embedder_instances:
        _check_search_deps()
        import warnings  # noqa: PLC0415

        from fastembed import TextEmbedding  # noqa: PLC0415

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _embedder_instances[model] = TextEmbedding(
                model_name=EMBEDDING_MODELS[model].fastembed_model
            )
    return _embedder_instances[model]


def _embed_query(model: str, query: str) -> Any:
    """Embed a query with the model's query prefix applied."""
    import numpy as np  # noqa: PLC0415

    prefixed = f"{EMBEDDING_MODELS[model].query_prefix}{query}"
    embedder = _get_embedder(model)
    return np.array(next(iter(embedder.embed([prefixed]))), dtype=np.float32)


def _load_search_index(model: str) -> dict[str, Any]:
    """Load and cache the search index for one embedding model.

    The index contains pre-normalized embedding vectors for fast cosine
    similarity via a single matrix-vector dot product.
    """
    import numpy as np  # noqa: PLC0415

    version = _get_icons_version()
    cached_index = _search_indexes.get(model)
    if cached_index is not None and cached_index["version"] == version:
        return cached_index

    db_path = _resolve_search_db(allow_download=True)
    if db_path is None:
        raise SearchNotAvailableError(
            "Search database not found. "
            "Build search data locally with: lucide build-search"
        )

    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT e.name, e.embedding, d.description "
            "FROM icon_embeddings e "
            "JOIN icon_descriptions d ON e.name = d.name "
            "WHERE e.model = ? "
            "ORDER BY e.name",
            (model,),
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    if not rows:
        raise SearchNotAvailableError(
            f"Search database contains no embeddings for model {model!r}"
        )

    names = [row[0] for row in rows]
    descriptions = {row[0]: row[2] for row in rows}
    matrix = np.stack([np.frombuffer(row[1], dtype=np.float32) for row in rows])

    # Pre-normalize so cosine similarity is a plain dot product
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    matrix = (matrix / norms).astype(np.float32)

    _search_indexes[model] = {
        "version": version,
        "names": names,
        "matrix": matrix,
        "descriptions": descriptions,
    }
    return _search_indexes[model]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search_icons(
    query: str,
    limit: int = 10,
    threshold: float = 0.0,
    model: str = DEFAULT_SEARCH_MODEL_ID,
) -> list[SearchResult]:
    """Search for icons by natural language query.

    On the first call the search database is downloaded and cached in
    ``~/.cache/python-lucide/``.  The embedding model (~35-130 MB depending
    on *model*) is also downloaded once by *fastembed*.

    Args:
        query: Natural language search query (e.g. "payment", "hard work").
        limit: Maximum number of results to return.
        threshold: Minimum cosine-similarity score (0.0-1.0) to include.
        model: Embedding model id from ``EMBEDDING_MODELS``.

    Returns:
        Results ordered by descending similarity score.

    Raises:
        ImportError: If the ``search`` extra is not installed.
        ValueError: If *model* is not a known embedding model id.
        SearchNotAvailableError: If search data cannot be loaded.
    """
    _check_search_deps()
    import numpy as np  # noqa: PLC0415

    model = _resolve_model(model)
    index = _load_search_index(model)

    query_vec = _embed_query(model, query)
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
