import pathlib
import sqlite3
import subprocess
from unittest import mock

import numpy as np
import pytest

from lucide import cli, search
from lucide.config import DEFAULT_EMBEDDING_DIM


@pytest.fixture
def temp_output_path(tmp_path):
    """Create a temporary path for the output database."""
    return tmp_path / "test-output.db"


def test_download_and_build_db_basic(temp_output_path):
    """Test basic functionality of the download_and_build_db function."""
    with (
        mock.patch("subprocess.run") as mock_run,
        mock.patch("pathlib.Path.exists", return_value=True),
        mock.patch("sqlite3.connect") as mock_connect,
        mock.patch("pathlib.Path.glob") as mock_glob,
        mock.patch("builtins.open", mock.mock_open(read_data="<svg></svg>")),
    ):
        mock_glob.return_value = [
            pathlib.Path("icons/home.svg"),
            pathlib.Path("icons/settings.svg"),
        ]

        mock_conn = mock.MagicMock()
        mock_cursor = mock.MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [("home",), ("settings",)]

        result = cli.download_and_build_db(
            output_path=temp_output_path, tag="test-tag", verbose=True
        )

        mock_run.assert_called_once()
        git_args = mock_run.call_args.args[0]
        assert "git" in git_args
        assert "clone" in git_args
        assert "--branch=test-tag" in git_args

        assert mock_connect.called
        assert mock_cursor.execute.called
        assert result == temp_output_path


def test_download_and_build_db_with_icon_list(temp_output_path):
    """Test building a database with a specific list of icons."""
    with (
        mock.patch("subprocess.run") as mock_run,
        mock.patch("pathlib.Path.exists", return_value=True),
        mock.patch("sqlite3.connect") as mock_connect,
        mock.patch("pathlib.Path.glob") as mock_glob,
        mock.patch("builtins.open", mock.mock_open(read_data="<svg></svg>")),
    ):
        mock_glob.return_value = [
            pathlib.Path("icons/home.svg"),
            pathlib.Path("icons/settings.svg"),
            pathlib.Path("icons/user.svg"),
        ]

        mock_conn = mock.MagicMock()
        mock_cursor = mock.MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [("home",)]

        result = cli.download_and_build_db(
            output_path=temp_output_path,
            tag="test-tag",
            icon_list=["home"],
            verbose=True,
        )

        assert mock_run.called
        assert mock_connect.called

        insert_calls = [
            call
            for call in mock_cursor.execute.mock_calls
            if "INSERT INTO" in str(call)
        ]
        assert len(insert_calls) > 0
        assert result == temp_output_path


def test_download_and_build_db_git_failure(temp_output_path):
    """Test behavior when git clone fails."""
    with mock.patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "git", stderr=b"error"),
    ):
        result = cli.download_and_build_db(output_path=temp_output_path, tag="test-tag")
        assert result is None


class TestMainSubcommands:
    def test_no_command_returns_error(self):
        with mock.patch("sys.argv", ["lucide"]):
            result = cli.main()
        assert result == 1

    def test_db_subcommand(self):
        with mock.patch("lucide.cli.download_and_build_db") as mock_build:
            mock_build.return_value = pathlib.Path("test.db")
            with mock.patch(
                "sys.argv",
                ["lucide", "db", "-o", "test.db", "-t", "test-tag", "-v"],
            ):
                result = cli.main()

            mock_build.assert_called_once_with(
                output_path="test.db",
                tag="test-tag",
                icon_list=None,
                icon_file=None,
                verbose=True,
            )
            assert result == 0

    def test_db_subcommand_with_icons(self):
        with mock.patch("lucide.cli.download_and_build_db") as mock_build:
            mock_build.return_value = pathlib.Path("test.db")
            with mock.patch(
                "sys.argv",
                ["lucide", "db", "-o", "test.db", "-i", "heart,star"],
            ):
                result = cli.main()

            mock_build.assert_called_once()
            call_kwargs = mock_build.call_args.kwargs
            assert call_kwargs["icon_list"] == ["heart", "star"]
            assert result == 0

    def test_version_subcommand(self):
        with mock.patch("lucide.dev_utils.print_version_status") as mock_version:
            with mock.patch("sys.argv", ["lucide", "version"]):
                result = cli.main()
            mock_version.assert_called_once()
            assert result == 0

    def test_legacy_lucide_db_alias(self):
        with mock.patch("lucide.cli.download_and_build_db") as mock_build:
            mock_build.return_value = pathlib.Path("test.db")
            with mock.patch("sys.argv", ["lucide-db", "-o", "test.db", "-v"]):
                result = cli.main_legacy_db()

            mock_build.assert_called_once()
            assert result == 0

    def test_search_subcommand(self, tmp_path):
        search_db = tmp_path / "test-search.db"
        conn = sqlite3.connect(search_db)
        conn.execute(
            "CREATE TABLE icon_descriptions"
            " (name TEXT PRIMARY KEY, description TEXT, model TEXT)"
        )
        conn.execute(
            "CREATE TABLE icon_embeddings"
            " (name TEXT PRIMARY KEY, embedding BLOB, model TEXT)"
        )
        conn.execute(
            "CREATE TABLE icon_clusters (name TEXT, cluster_id INTEGER, theme TEXT)"
        )
        conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        emb = (
            np.random.default_rng(42)
            .standard_normal(DEFAULT_EMBEDDING_DIM)
            .astype(np.float32)
        )
        conn.execute(
            "INSERT INTO icon_descriptions VALUES (?, ?, ?)",
            ("heart", "A heart icon", "test"),
        )
        conn.execute(
            "INSERT INTO icon_embeddings VALUES (?, ?, ?)",
            ("heart", emb.tobytes(), "test"),
        )
        conn.execute("INSERT INTO metadata VALUES ('version', 'test')")
        conn.commit()
        conn.close()

        with mock.patch(
            "sys.argv",
            ["lucide", "search", "love", "--search-db", str(search_db)],
        ):
            result = cli.main()
        assert result == 0

    def test_search_subcommand_no_db(self, monkeypatch):
        monkeypatch.delenv("LUCIDE_SEARCH_DB_PATH", raising=False)
        search._search_index = None
        search._embedder_instance = None

        with mock.patch(
            "sys.argv",
            [
                "lucide",
                "search",
                "love",
                "--search-db",
                "/nonexistent/search.db",
            ],
        ):
            result = cli.main()
        assert result == 1
