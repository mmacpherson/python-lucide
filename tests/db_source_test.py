import contextlib
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from build_support import SQL_SOURCE, export_database, restore_database

ROOT = Path(__file__).resolve().parents[1]


def test_export_is_stable_across_row_order_and_build_timestamps(tmp_path):
    dumps = []
    for index, rows in enumerate(
        (
            [("b", "Bob's icon"), ("a", "first\nline")],
            [("a", "first\nline"), ("b", "Bob's icon")],
        )
    ):
        database = tmp_path / f"{index}.db"
        with contextlib.closing(sqlite3.connect(database)) as conn, conn:
            conn.executescript(
                "CREATE TABLE icons(name TEXT PRIMARY KEY, svg TEXT NOT NULL);"
                "CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);"
            )
            conn.executemany("INSERT INTO icons VALUES (?, ?)", rows)
            conn.executemany(
                "INSERT INTO metadata VALUES (?, ?)",
                [("version", "test"), ("created_at", str(index))],
            )
        source = tmp_path / f"{index}.sql"
        export_database(database, source)
        dumps.append(source.read_bytes())
    assert dumps[0] == dumps[1]
    assert b"created_at" not in dumps[0]
    assert b"Bob''s icon" in dumps[0]


def test_bundled_source_round_trips_exactly(tmp_path):
    database = tmp_path / "icons.db"
    restore_database(SQL_SOURCE, database)
    exported = tmp_path / "icons.sql"
    export_database(database, exported)
    assert exported.read_bytes() == SQL_SOURCE.read_bytes()
    with contextlib.closing(sqlite3.connect(database)) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert conn.execute("SELECT count(*) FROM icons").fetchone()[0] > 1800
        assert conn.execute(
            "SELECT name, deprecated, deprecation_reason FROM icon_aliases "
            "WHERE alias='album'"
        ).fetchone() == ("square-bookmark", 1, "alias.name")


def test_restored_database_rejects_ambiguous_aliases(tmp_path):
    database = tmp_path / "icons.db"
    restore_database(SQL_SOURCE, database)
    with (
        contextlib.closing(sqlite3.connect(database)) as conn,
        pytest.raises(sqlite3.IntegrityError, match=r"icon_aliases\.alias"),
    ):
        conn.execute("INSERT INTO icon_aliases(name, alias) VALUES('circle', 'album')")


@pytest.mark.parametrize(
    ("table", "column", "value"),
    [("icon_tags", "tag", "book"), ("icon_categories", "category", "text")],
)
def test_reverse_metadata_lookups_use_covering_indexes(tmp_path, table, column, value):
    database = tmp_path / "icons.db"
    restore_database(SQL_SOURCE, database)
    with contextlib.closing(sqlite3.connect(database)) as conn:
        query = f"SELECT name FROM {table} WHERE {column} = ? ORDER BY name"
        assert conn.execute(query, (value,)).fetchall()
        plan = conn.execute("EXPLAIN QUERY PLAN " + query, (value,)).fetchall()
        details = " ".join(row[3] for row in plan)
        assert "USING COVERING INDEX" in details
        assert "TEMP B-TREE" not in details


@pytest.mark.parametrize("corruption", ["syntax", "empty", "dangling", "version"])
def test_invalid_source_preserves_previous_database(tmp_path, corruption):
    database = tmp_path / "icons.db"
    database.write_bytes(b"previous database")
    source = tmp_path / "bad.sql"
    suffix = {
        "syntax": "invalid sql;",
        "empty": "DELETE FROM icons;",
        "dangling": "INSERT INTO icon_aliases(name,alias) VALUES('missing','bad');",
        "version": "DELETE FROM metadata WHERE key='version';",
    }[corruption]
    source.write_text(SQL_SOURCE.read_text() + suffix)
    with pytest.raises((sqlite3.Error, ValueError)):
        restore_database(source, database)
    assert database.read_bytes() == b"previous database"
    assert list(tmp_path.glob("*.db")) == [database]


def test_clean_source_build_produces_installable_wheel_and_rebuildable_sdist(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    for name in (
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "Makefile",
        "hatch_build.py",
        "build_support.py",
    ):
        shutil.copyfile(ROOT / name, checkout / name)
    shutil.copytree(
        ROOT / "src",
        checkout / "src",
        ignore=shutil.ignore_patterns("*.db", "__pycache__", "*.pyc"),
    )
    assert not list(checkout.rglob("*.db"))
    subprocess.run(
        ["uv", "build", "--no-sources", "--python", sys.executable],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    sdist = next((checkout / "dist").glob("*.tar.gz"))
    with tarfile.open(sdist) as archive:
        names = archive.getnames()
        assert any(name.endswith("/lucide-icons.sql") for name in names)
        assert any(name.endswith("/build_support.py") for name in names)
        assert not any(name.endswith(".db") for name in names)
    # uv build creates the wheel from the sdist, exercising the source-only path.
    wheel = next((checkout / "dist").glob("*.whl"))
    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "lucide/data/lucide-icons.db" in names
        assert not any(
            name.endswith(
                (".sql", "lucide-search.db", ".jsonl", "lucide-icon-clusters.json")
            )
            for name in names
        )
        archive.extractall(installed)
    probe = """
import sys, warnings
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import lucide
assert Path(lucide.__file__).is_relative_to(Path(sys.argv[1]))
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    svg = lucide.lucide_icon('album')
assert 'lucide-placeholder' not in svg
assert len(caught) == 1
assert caught[0].category is DeprecationWarning
assert "use 'square-bookmark' instead" in str(caught[0].message)
"""
    subprocess.run(
        [sys.executable, "-I", "-c", probe, str(installed)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
