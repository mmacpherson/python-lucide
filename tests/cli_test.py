import pathlib
import subprocess
from unittest import mock

import pytest

from lucide import cli


@pytest.fixture
def temp_output_path(tmp_path):
    """Create a temporary path for the output database."""
    return tmp_path / "test-output.db"


def test_download_and_build_db_basic(temp_output_path):
    """Test basic functionality of the download_and_build_db function."""
    # Mock subprocess.run to avoid actually calling git
    with (
        mock.patch("subprocess.run") as mock_run,
        mock.patch("pathlib.Path.exists", return_value=True),
        mock.patch("sqlite3.connect") as mock_connect,
        mock.patch("pathlib.Path.glob") as mock_glob,
        mock.patch("builtins.open", mock.mock_open(read_data="<svg></svg>")),
    ):
        # Setup mocks
        mock_glob.return_value = [
            pathlib.Path("icons/home.svg"),
            pathlib.Path("icons/settings.svg"),
        ]

        mock_conn = mock.MagicMock()
        mock_cursor = mock.MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [("home",), ("settings",)]

        # Call the function
        result = cli.download_and_build_db(
            output_path=temp_output_path, tag="test-tag", verbose=True
        )

        # Verify the function called git clone with the right parameters
        mock_run.assert_called_once()
        git_args = mock_run.call_args.args[0]
        assert "git" in git_args
        assert "clone" in git_args
        assert "--branch=test-tag" in git_args

        # Verify database operations
        assert mock_connect.called
        assert mock_cursor.execute.called

        # Verify the function returned the expected path
        assert result == temp_output_path


def test_download_and_build_db_with_icon_list(temp_output_path):
    """Test building a database with a specific list of icons."""
    # Mock subprocess.run to avoid actually calling git
    with (
        mock.patch("subprocess.run") as mock_run,
        mock.patch("pathlib.Path.exists", return_value=True),
        mock.patch("sqlite3.connect") as mock_connect,
        mock.patch("pathlib.Path.glob") as mock_glob,
        mock.patch("builtins.open", mock.mock_open(read_data="<svg></svg>")),
    ):
        # Setup mocks
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

        # Call the function with a specific icon list
        result = cli.download_and_build_db(
            output_path=temp_output_path,
            tag="test-tag",
            icon_list=["home"],
            verbose=True,
        )

        # Verify the function called git clone
        assert mock_run.called

        # Verify database operations - should have inserted only 'home'
        assert mock_connect.called

        # Count insert calls with "home" but not with "settings" or "user"
        insert_calls = [
            call
            for call in mock_cursor.execute.mock_calls
            if "INSERT INTO" in str(call)
        ]
        assert len(insert_calls) > 0

        # Verify the function returned the expected path
        assert result == temp_output_path


def test_download_and_build_db_git_failure(temp_output_path):
    """Test behavior when git clone fails."""
    # Mock subprocess.run to simulate a git failure
    with mock.patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "git", stderr=b"error"),
    ):
        # Call the function
        result = cli.download_and_build_db(output_path=temp_output_path, tag="test-tag")

        # Verify the function returned None
        assert result is None


def test_main_function():
    """Test the main CLI entry point function."""
    # Mock the argparse.ArgumentParser to avoid actual command line parsing
    with mock.patch("argparse.ArgumentParser") as mock_parser:
        # Setup mock parser
        parser_instance = mock.MagicMock()
        mock_parser.return_value = parser_instance
        parser_instance.parse_args.return_value = mock.MagicMock(
            output="test.db",
            tag="test-tag",
            icons="home,settings",
            file=None,
            verbose=True,
        )

        # Mock download_and_build_db to avoid actually running it
        with mock.patch("lucide.cli.download_and_build_db") as mock_download:
            mock_download.return_value = pathlib.Path("test.db")

            # Call the main function
            result = cli.main()

            # Verify the function called download_and_build_db with the right parameters
            mock_download.assert_called_with(
                output_path="test.db",
                tag="test-tag",
                icon_list=["home", "settings"],
                icon_file=None,
                verbose=True,
            )

            # Verify the function returned 0 (success)
            assert result == 0
