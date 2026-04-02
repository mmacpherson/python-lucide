"""Tests for the build-time search data pipeline."""

import json
import sqlite3

import numpy as np
import pytest

from lucide import build_search
from lucide.config import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VLM_MODEL,
)

# --- Fixtures ---


@pytest.fixture
def main_db(tmp_path):
    """Create a temporary main icons database with test icons."""
    db_path = tmp_path / "lucide-icons.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE icons (name TEXT PRIMARY KEY, svg TEXT NOT NULL)")
    cursor.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    cursor.execute("INSERT INTO metadata VALUES ('version', '0.577.0')")

    xmlns = "http://www.w3.org/2000/svg"
    for name in ("heart", "star", "circle"):
        svg = (
            f'<svg xmlns="{xmlns}" width="24" height="24" viewBox="0 0 24 24">'
            f"<circle cx='12' cy='12' r='10'></circle></svg>"
        )
        cursor.execute("INSERT INTO icons VALUES (?, ?)", (name, svg))

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def search_db(tmp_path):
    """Return a path for the search database (not yet created)."""
    return tmp_path / "lucide-search.db"


@pytest.fixture
def jsonl_path(tmp_path):
    """Return a path for the descriptions JSONL (not yet created)."""
    return tmp_path / "gemini-icon-descriptions.jsonl"


@pytest.fixture
def icons_dir(tmp_path):
    """Create a mock Lucide icons directory with JSON metadata."""
    d = tmp_path / "icons"
    d.mkdir()

    for name, tags, categories in [
        ("heart", ["love", "emotion"], ["social", "emoji"]),
        ("star", ["favorite", "rating"], ["social"]),
        ("circle", ["shape", "round"], ["shapes"]),
    ]:
        meta = {
            "$schema": "../icon.schema.json",
            "tags": tags,
            "categories": categories,
        }
        (d / f"{name}.json").write_text(json.dumps(meta))

    return d


@pytest.fixture
def sample_jsonl(jsonl_path):
    """Create a JSONL file with sample descriptions."""
    for name in ("heart", "star", "circle"):
        record = {
            "name": name,
            "description": f"A {name} icon for testing",
            "model": DEFAULT_VLM_MODEL,
            "prompt_template_hash": "test",
            "lucide_version": "0.577.0",
            "tags": [],
            "categories": [],
            "timestamp": "2026-03-25T00:00:00+00:00",
        }
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")
    return jsonl_path


# --- Tests ---


class TestEnsureSearchTables:
    def test_creates_tables(self, search_db):
        build_search.ensure_search_tables(search_db)

        conn = sqlite3.connect(search_db)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()

        assert "icon_descriptions" in tables
        assert "icon_embeddings" in tables
        assert "metadata" in tables

    def test_idempotent(self, search_db):
        build_search.ensure_search_tables(search_db)
        build_search.ensure_search_tables(search_db)

        conn = sqlite3.connect(search_db)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert "icon_descriptions" in tables


class TestReadIconMetadata:
    def test_reads_tags_and_categories(self, icons_dir):
        meta = build_search._read_icon_metadata(icons_dir, "heart")
        assert meta["tags"] == ["love", "emotion"]
        assert meta["categories"] == ["social", "emoji"]

    def test_missing_json_returns_empty(self, icons_dir):
        meta = build_search._read_icon_metadata(icons_dir, "nonexistent")
        assert meta["tags"] == []
        assert meta["categories"] == []


class TestBuildMetadataLine:
    def test_with_metadata(self, icons_dir):
        line = build_search._build_metadata_line(icons_dir, "heart")
        assert "love" in line
        assert "emotion" in line
        assert "social" in line

    def test_without_icons_dir(self):
        line = build_search._build_metadata_line(None, "heart")
        assert line == ""


class TestLoadDescriptionsJsonl:
    def test_loads_records(self, sample_jsonl):
        records = build_search.load_descriptions_jsonl(sample_jsonl)
        assert set(records.keys()) == {"heart", "star", "circle"}
        assert records["heart"]["description"] == "A heart icon for testing"
        assert records["heart"]["model"] == DEFAULT_VLM_MODEL

    def test_empty_file(self, jsonl_path):
        jsonl_path.write_text("")
        records = build_search.load_descriptions_jsonl(jsonl_path)
        assert records == {}


class TestGenerateDescriptions:
    def test_requires_api_key(self, main_db, jsonl_path, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Gemini API key"):
            build_search.generate_descriptions(main_db, jsonl_path)

    def test_writes_jsonl(self, main_db, jsonl_path, icons_dir, monkeypatch):
        monkeypatch.setattr(
            build_search,
            "_call_gemini",
            lambda prompt, image, key, **kw: "Description for icon",
        )
        monkeypatch.setattr(
            build_search,
            "_render_svg_to_png",
            lambda svg, **kw: b"\x89PNG fake data",
        )

        count = build_search.generate_descriptions(
            main_db,
            jsonl_path,
            icons_dir=icons_dir,
            api_key="fake-key",
        )

        assert count == 3
        assert jsonl_path.exists()

        records = build_search.load_descriptions_jsonl(jsonl_path)
        assert len(records) == 3
        assert all(r["model"] == DEFAULT_VLM_MODEL for r in records.values())
        assert records["heart"]["tags"] == ["love", "emotion"]
        assert records["heart"]["lucide_version"] == "0.577.0"

    def test_incremental_skips_existing(self, main_db, sample_jsonl, monkeypatch):
        monkeypatch.setattr(
            build_search,
            "_call_gemini",
            lambda prompt, image, key, **kw: "New description",
        )
        monkeypatch.setattr(
            build_search,
            "_render_svg_to_png",
            lambda svg, **kw: b"\x89PNG fake data",
        )

        # sample_jsonl already has heart, star, circle
        count = build_search.generate_descriptions(
            main_db,
            sample_jsonl,
            api_key="fake-key",
            incremental=True,
        )
        assert count == 0  # All already exist


@pytest.fixture
def sample_clusters(tmp_path):
    """Create a clusters JSON file."""
    data = {
        "clusters": {
            "0": {"icons": ["heart", "star"], "theme": "Love & Favorites"},
            "1": {"icons": ["circle"], "theme": "Shapes"},
            "-1": {"icons": [], "theme": "Unclustered"},
        },
        "generated_at": "2026-03-26T00:00:00+00:00",
        "naming_model": "test",
    }
    path = tmp_path / "clusters.json"
    path.write_text(json.dumps(data))
    return path


class TestBuildSearchDb:
    def test_builds_db_from_jsonl(self, sample_jsonl, search_db, sample_clusters):
        build_search.build_search_db(sample_jsonl, search_db, sample_clusters)

        conn = sqlite3.connect(search_db)

        # Check descriptions
        desc_rows = conn.execute(
            "SELECT name, description FROM icon_descriptions ORDER BY name"
        ).fetchall()
        assert len(desc_rows) == 3
        assert desc_rows[0][0] == "circle"

        # Check embeddings
        emb_rows = conn.execute(
            "SELECT name, embedding, model FROM icon_embeddings"
        ).fetchall()
        assert len(emb_rows) == 3
        for row in emb_rows:
            emb = np.frombuffer(row[1], dtype=np.float32)
            assert emb.shape == (DEFAULT_EMBEDDING_DIM,)
            assert row[2] == DEFAULT_EMBEDDING_MODEL

        # Check metadata
        meta = dict(conn.execute("SELECT key, value FROM metadata").fetchall())
        assert meta["version"] == "0.577.0"
        assert meta["embedding_model"] == DEFAULT_EMBEDDING_MODEL
        assert meta["embedding_dim"] == str(DEFAULT_EMBEDDING_DIM)
        assert "built_at" in meta

        # Check clusters were loaded
        cluster_rows = conn.execute(
            "SELECT name, theme FROM icon_clusters ORDER BY name"
        ).fetchall()
        assert len(cluster_rows) == 3
        themes = {r[1] for r in cluster_rows}
        assert "Love & Favorites" in themes
        assert "Shapes" in themes

        conn.close()

    def test_rebuilds_from_scratch(self, sample_jsonl, search_db, sample_clusters):
        build_search.build_search_db(sample_jsonl, search_db, sample_clusters)

        # Build again — should replace, not duplicate
        build_search.build_search_db(sample_jsonl, search_db, sample_clusters)

        conn = sqlite3.connect(search_db)
        count = conn.execute("SELECT COUNT(*) FROM icon_descriptions").fetchone()[0]
        conn.close()
        assert count == 3
