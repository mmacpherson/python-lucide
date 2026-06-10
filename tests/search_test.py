"""Tests for the semantic search module."""

import sqlite3
from unittest import mock

import numpy as np
import pytest

from lucide import search
from lucide.config import DEFAULT_EMBEDDING_DIM

# --- Fixtures ---


@pytest.fixture(autouse=True)
def _reset_search_caches():
    """Clear module-level caches between tests."""
    search._embedder_instance = None
    search._search_index = None
    yield
    search._embedder_instance = None
    search._search_index = None


@pytest.fixture
def search_db(tmp_path):
    """Create a temporary search database with test data."""
    db_path = tmp_path / "lucide-search.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "CREATE TABLE icon_descriptions ("
        "  name TEXT PRIMARY KEY, description TEXT NOT NULL, model TEXT NOT NULL"
        ")"
    )
    cursor.execute(
        "CREATE TABLE icon_embeddings ("
        "  name TEXT PRIMARY KEY, embedding BLOB NOT NULL, model TEXT NOT NULL"
        ")"
    )
    cursor.execute(
        "CREATE TABLE metadata (  key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )

    # Insert test data with known embeddings for deterministic similarity tests
    icons = [
        ("heart", "A heart shape representing love, affection, and emotion."),
        ("credit-card", "A credit card for payment, purchase, and transactions."),
        ("bird", "A bird in flight representing freedom, nature, and wildlife."),
    ]

    rng = np.random.default_rng(42)
    embeddings = {}
    for name, desc in icons:
        cursor.execute(
            "INSERT INTO icon_descriptions VALUES (?, ?, ?)",
            (name, desc, "test-model"),
        )
        emb = rng.standard_normal(DEFAULT_EMBEDDING_DIM).astype(np.float32)
        emb /= np.linalg.norm(emb)
        embeddings[name] = emb
        cursor.execute(
            "INSERT INTO icon_embeddings VALUES (?, ?, ?)",
            (name, emb.tobytes(), "test-model"),
        )

    cursor.execute("INSERT INTO metadata VALUES ('version', '0.577.0')")
    conn.commit()
    conn.close()

    return db_path, embeddings


@pytest.fixture
def mock_main_db(tmp_path):
    """Create a temporary main icons database."""
    db_path = tmp_path / "lucide-icons.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE icons (name TEXT PRIMARY KEY, svg TEXT NOT NULL)")
    cursor.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    cursor.execute("INSERT INTO metadata VALUES ('version', '0.577.0')")
    cursor.execute("INSERT INTO icons VALUES ('heart', '<svg>heart</svg>')")
    conn.commit()
    conn.close()
    return db_path


class FakeEmbedder:
    """Mock fastembed TextEmbedding that returns deterministic vectors."""

    def __init__(self, **_kwargs):
        self.rng = np.random.default_rng(99)

    def embed(self, texts):
        for _text in texts:
            vec = self.rng.standard_normal(DEFAULT_EMBEDDING_DIM).astype(np.float32)
            vec /= np.linalg.norm(vec)
            yield vec


# --- Tests ---


class TestSearchIcons:
    def test_returns_ranked_results(self, search_db, monkeypatch):
        db_path, _embeddings = search_db

        monkeypatch.setenv("LUCIDE_SEARCH_DB_PATH", str(db_path))
        monkeypatch.setattr(search, "_get_icons_version", lambda: "0.577.0")

        with mock.patch.dict("sys.modules", {"fastembed": mock.MagicMock()}):
            search._embedder_instance = FakeEmbedder()

            results = search.search_icons("love", limit=3)

        assert len(results) > 0
        assert all(isinstance(r, search.SearchResult) for r in results)
        # Results should be sorted by score descending
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_limit_parameter(self, search_db, monkeypatch):
        db_path, _ = search_db

        monkeypatch.setenv("LUCIDE_SEARCH_DB_PATH", str(db_path))
        monkeypatch.setattr(search, "_get_icons_version", lambda: "0.577.0")

        with mock.patch.dict("sys.modules", {"fastembed": mock.MagicMock()}):
            search._embedder_instance = FakeEmbedder()

            results = search.search_icons("test", limit=1)

        assert len(results) == 1

    def test_threshold_filters_low_scores(self, search_db, monkeypatch):
        db_path, _ = search_db

        monkeypatch.setenv("LUCIDE_SEARCH_DB_PATH", str(db_path))
        monkeypatch.setattr(search, "_get_icons_version", lambda: "0.577.0")

        with mock.patch.dict("sys.modules", {"fastembed": mock.MagicMock()}):
            search._embedder_instance = FakeEmbedder()

            # Very high threshold should exclude most results
            results = search.search_icons("test", threshold=0.99)

        assert len(results) == 0

    def test_result_has_description(self, search_db, monkeypatch):
        db_path, _ = search_db

        monkeypatch.setenv("LUCIDE_SEARCH_DB_PATH", str(db_path))
        monkeypatch.setattr(search, "_get_icons_version", lambda: "0.577.0")

        with mock.patch.dict("sys.modules", {"fastembed": mock.MagicMock()}):
            search._embedder_instance = FakeEmbedder()

            results = search.search_icons("heart", limit=3)

        for r in results:
            assert r.description  # non-empty
            assert r.name  # non-empty


class TestSearchAvailable:
    def test_returns_true_when_db_exists(self, search_db, monkeypatch):
        db_path, _ = search_db
        monkeypatch.setenv("LUCIDE_SEARCH_DB_PATH", str(db_path))
        assert search.search_available() is True

    def test_returns_false_when_no_db(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LUCIDE_SEARCH_DB_PATH", str(tmp_path / "nonexistent.db"))
        assert search.search_available() is False

    def test_returns_false_for_empty_db(self, tmp_path, monkeypatch):
        db_path = tmp_path / "empty-search.db"
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE icon_embeddings "
            "(name TEXT PRIMARY KEY, embedding BLOB NOT NULL, model TEXT NOT NULL)"
        )
        conn.commit()
        conn.close()

        monkeypatch.setenv("LUCIDE_SEARCH_DB_PATH", str(db_path))
        assert search.search_available() is False


class TestGetIconDescription:
    def test_returns_description(self, search_db, monkeypatch):
        db_path, _ = search_db
        monkeypatch.setenv("LUCIDE_SEARCH_DB_PATH", str(db_path))

        desc = search.get_icon_description("heart")
        assert desc is not None
        assert "heart" in desc.lower() or "love" in desc.lower()

    def test_returns_none_for_missing_icon(self, search_db, monkeypatch):
        db_path, _ = search_db
        monkeypatch.setenv("LUCIDE_SEARCH_DB_PATH", str(db_path))

        desc = search.get_icon_description("nonexistent-icon")
        assert desc is None

    def test_returns_none_when_no_db(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LUCIDE_SEARCH_DB_PATH", str(tmp_path / "nope.db"))
        assert search.get_icon_description("heart") is None


class TestSearchNotAvailableError:
    def test_raised_when_db_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LUCIDE_SEARCH_DB_PATH", str(tmp_path / "nope.db"))
        monkeypatch.delenv("LUCIDE_SEARCH_DB_PATH", raising=False)
        monkeypatch.setattr(search, "_get_icons_version", lambda: "0.577.0")

        # Mock the cache dir to return a path with no cached file
        monkeypatch.setattr(search, "_get_cache_dir", lambda: tmp_path / "cache")
        (tmp_path / "cache").mkdir()

        # Mock urllib to simulate download failure
        with (
            mock.patch.object(
                search.urllib.request,
                "urlretrieve",
                side_effect=Exception("404"),
            ),
            mock.patch.dict("sys.modules", {"fastembed": mock.MagicMock()}),
        ):
            search._embedder_instance = FakeEmbedder()
            with pytest.raises(search.SearchNotAvailableError):
                search.search_icons("test")


class TestCheckSearchDeps:
    def test_import_error_message(self):
        with (
            mock.patch.dict("sys.modules", {"fastembed": None}),
            pytest.raises(ImportError, match="python-lucide\\[search\\]"),
        ):
            search._check_search_deps()


class TestCosineSimilarity:
    """Verify the cosine similarity math with known vectors."""

    def test_identical_vectors_score_one(self):
        vec = np.array([1, 0, 0], dtype=np.float32)
        matrix = vec.reshape(1, -1)
        scores = matrix @ vec
        assert abs(float(scores[0]) - 1.0) < 1e-6

    def test_orthogonal_vectors_score_zero(self):
        query = np.array([1, 0, 0], dtype=np.float32)
        matrix = np.array([[0, 1, 0]], dtype=np.float32)
        scores = matrix @ query
        assert abs(float(scores[0])) < 1e-6

    def test_opposite_vectors_score_negative(self):
        query = np.array([1, 0, 0], dtype=np.float32)
        matrix = np.array([[-1, 0, 0]], dtype=np.float32)
        scores = matrix @ query
        assert float(scores[0]) < 0


class TestResolveSearchDb:
    def test_env_var_takes_priority(self, search_db, monkeypatch):
        db_path, _ = search_db
        monkeypatch.setenv("LUCIDE_SEARCH_DB_PATH", str(db_path))

        result = search._resolve_search_db(allow_download=False)
        assert result == db_path

    def test_returns_none_when_not_found(self, tmp_path, monkeypatch):
        monkeypatch.delenv("LUCIDE_SEARCH_DB_PATH", raising=False)
        monkeypatch.setattr(search, "_get_cache_dir", lambda: tmp_path / "empty-cache")
        (tmp_path / "empty-cache").mkdir()
        monkeypatch.setattr(search, "_get_icons_version", lambda: "0.577.0")

        result = search._resolve_search_db(allow_download=False)
        assert result is None

    def test_cached_file_found(self, search_db, tmp_path, monkeypatch):
        db_path, _ = search_db
        monkeypatch.delenv("LUCIDE_SEARCH_DB_PATH", raising=False)

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cached = cache_dir / "lucide-search-0.577.0.db"
        # Copy the test DB to the cache location
        cached.write_bytes(db_path.read_bytes())

        monkeypatch.setattr(search, "_get_cache_dir", lambda: cache_dir)
        monkeypatch.setattr(search, "_get_icons_version", lambda: "0.577.0")

        result = search._resolve_search_db(allow_download=False)
        assert result == cached
