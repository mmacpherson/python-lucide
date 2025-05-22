#!/usr/bin/env python3
"""Database utilities for working with the Lucide icons database."""

import contextlib
import importlib.resources
import logging
import os
import pathlib
import sqlite3
from collections.abc import Generator
from typing import Protocol


# Define a protocol for importlib.resources.abc.Traversable for type checking
class TraversableResource(Protocol):
    """Protocol for importlib.resources.abc.Traversable for type checking."""

    def joinpath(self, *paths: str) -> "TraversableResource":
        """Join the resource path with the given paths."""
        ...

    def __str__(self) -> str:
        """Return the string representation of the resource path."""
        ...


logger = logging.getLogger(__name__)


def get_default_db_path() -> pathlib.Path | None:
    """Determines the default path for the Lucide icons database.

    The function looks for the database in the following locations (in order):
    1. The path specified in the LUCIDE_PY_DB_PATH environment variable
    2. In the package data directory (if installed with the 'db' extra)
    3. In the current working directory

    Returns:
        pathlib.Path: The path to the database file
    """
    # First try environment variable
    if "LUCIDE_DB_PATH" in os.environ:
        return pathlib.Path(os.environ["LUCIDE_DB_PATH"])

    # Then try package data
    try:
        package_data = importlib.resources.files("lucide.data")
        db_file_path = package_data.joinpath("lucide-icons.db")
        # Convert to string and then to Path to handle Traversable objects
        db_path = pathlib.Path(str(db_file_path))
        if db_path.exists():
            return db_path
    except (ModuleNotFoundError, FileNotFoundError, ImportError):
        pass

    # Finally, look in current directory
    return pathlib.Path.cwd() / "lucide-icons.db"


@contextlib.contextmanager
def get_db_connection(
    db_path: pathlib.Path | str | None = None,
) -> Generator[sqlite3.Connection | None, None, None]:
    """Provides a SQLite database connection as a context manager.

    Args:
        db_path: Path to the database file. If None, uses the default path.

    Yields:
        sqlite3.Connection or None: A database connection object or None if connection
        fails.
    """
    conn = None

    if db_path is None:
        db_path = get_default_db_path()

    # Convert string path to Path object
    if isinstance(db_path, str):
        db_path = pathlib.Path(db_path)

    # Handle None case
    if db_path is None:
        logger.error("No database path could be determined")
        yield None
        return

    try:
        if not db_path.exists():
            logger.error(f"Database file does not exist: {db_path}")
            yield None
            return

        # Read-only connection
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        yield conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        logger.error(f"Database path: {db_path}")
        yield None
    finally:
        if conn:
            conn.close()
