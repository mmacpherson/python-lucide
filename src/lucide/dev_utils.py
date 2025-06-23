#!/usr/bin/env python3
"""Development utilities for checking Lucide version updates and artifact status."""

import contextlib
import json
import pathlib
import sqlite3
import urllib.request
from datetime import datetime
from typing import NamedTuple

from .config import DEFAULT_LUCIDE_TAG
from .db import get_default_db_path


class VersionStatus(NamedTuple):
    """Status information about Lucide version and artifacts."""

    current_config_version: str
    latest_github_version: str | None
    database_exists: bool
    database_version: str | None
    database_created_at: datetime | None
    config_modified_at: datetime | None
    needs_update: bool
    recommendations: list[str]


def get_latest_lucide_version() -> str | None:
    """Fetch the latest Lucide version from GitHub API.

    Returns:
        Latest version tag or None if unable to fetch
    """
    try:
        url = "https://api.github.com/repos/lucide-icons/lucide/releases/latest"
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())
            tag_name = data.get("tag_name")
            return tag_name if isinstance(tag_name, str) else None
    except (urllib.error.URLError, json.JSONDecodeError, KeyError):
        return None


def get_database_metadata() -> tuple[str | None, datetime | None]:
    """Get version and creation timestamp from database metadata.

    Returns:
        Tuple of (version, created_at) or (None, None) if not available
    """
    db_path = get_default_db_path()
    if not db_path or not db_path.exists():
        return None, None

    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            cursor = conn.cursor()

            # Check if metadata table exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='metadata'
            """)
            if not cursor.fetchone():
                # No metadata table, fall back to file modification time
                stat = db_path.stat()
                return None, datetime.fromtimestamp(stat.st_mtime)

            # Get version and creation time from metadata
            cursor.execute("""
                SELECT key, value FROM metadata
                WHERE key IN ('version', 'created_at')
            """)
            metadata = dict(cursor.fetchall())

            version = metadata.get("version")
            created_at_str = metadata.get("created_at")
            created_at = None
            if created_at_str:
                with contextlib.suppress(ValueError):
                    created_at = datetime.fromisoformat(created_at_str)

            # If no creation time in metadata, use file time
            if not created_at:
                stat = db_path.stat()
                created_at = datetime.fromtimestamp(stat.st_mtime)

            return version, created_at

    except sqlite3.Error:
        return None, None


def get_config_modification_time() -> datetime | None:
    """Get the modification time of the config file.

    Returns:
        Modification datetime or None if not available
    """
    try:
        config_path = pathlib.Path(__file__).parent / "config.py"
        if config_path.exists():
            stat = config_path.stat()
            return datetime.fromtimestamp(stat.st_mtime)
    except OSError:
        pass
    return None


def check_version_status() -> VersionStatus:
    """Check the current status of Lucide version and artifacts.

    Returns:
        VersionStatus with detailed information and recommendations
    """
    current_version = DEFAULT_LUCIDE_TAG
    latest_version = get_latest_lucide_version()

    db_path = get_default_db_path()
    db_exists = db_path is not None and db_path.exists()
    db_version, db_created_at = get_database_metadata()
    config_modified_at = get_config_modification_time()

    recommendations = []
    needs_update = False

    # Check if database exists
    if not db_exists:
        needs_update = True
        recommendations.append("Database not found. Run 'make lucide-db' to create it.")
    else:
        # Check if database version matches config
        if db_version and db_version != current_version:
            needs_update = True
            recommendations.append(
                f"Database version ({db_version}) doesn't match config "
                f"({current_version}). Run 'make lucide-db' to rebuild."
            )
        elif not db_version:
            recommendations.append(
                "Database version unknown (no metadata). "
                "Consider rebuilding with 'make lucide-db'."
            )

        # Check if config was modified after database creation
        if config_modified_at and db_created_at and config_modified_at > db_created_at:
            needs_update = True
            recommendations.append(
                "Config file modified after database creation. "
                "Run 'make lucide-db' to rebuild."
            )

    # Check if there's a newer version available
    if latest_version and latest_version != current_version:
        recommendations.append(
            f"Newer Lucide version available: {latest_version} "
            f"(current: {current_version}). Consider updating "
            "DEFAULT_LUCIDE_TAG in config.py."
        )

    if not recommendations:
        recommendations.append("All artifacts are up to date!")

    return VersionStatus(
        current_config_version=current_version,
        latest_github_version=latest_version,
        database_exists=db_exists,
        database_version=db_version,
        database_created_at=db_created_at,
        config_modified_at=config_modified_at,
        needs_update=needs_update,
        recommendations=recommendations,
    )


def print_version_status() -> int:
    """Print version status information to console.

    Returns:
        Exit code: 0 if everything is up to date, 1 if updates needed
    """
    status = check_version_status()

    print("🔍 Lucide Version Check")
    print("=" * 50)
    print(f"Config version:    {status.current_config_version}")

    if status.latest_github_version:
        print(f"Latest on GitHub:  {status.latest_github_version}")
    else:
        print("Latest on GitHub:  Unable to fetch")

    print()
    print("📊 Database Status")
    print("-" * 20)
    if status.database_exists:
        print("Database exists:   ✅ Yes")
        if status.database_version:
            print(f"Database version:  {status.database_version}")
        else:
            print("Database version:  Unknown")

        if status.database_created_at:
            created_str = status.database_created_at.strftime("%Y-%m-%d %H:%M:%S")
            print(f"Created at:        {created_str}")
    else:
        print("Database exists:   ❌ No")

    if status.config_modified_at:
        modified_str = status.config_modified_at.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Config modified:   {modified_str}")

    print()
    print("💡 Recommendations")
    print("-" * 20)
    for rec in status.recommendations:
        icon = (
            "⚠️ " if status.needs_update and rec != status.recommendations[-1] else "✅ "
        )
        print(f"{icon} {rec}")

    return 1 if status.needs_update else 0


def compare_versions(current: str, latest: str | None) -> bool:
    """Compare current and latest versions to determine if update is needed.

    Args:
        current: Current version string
        latest: Latest version string or None if fetch failed

    Returns:
        True if update is needed, False otherwise
    """
    return latest is not None and current != latest


def get_icon_count_from_db(db_path: pathlib.Path | str | None = None) -> int:
    """Get the number of icons in the database.

    Args:
        db_path: Path to database file, uses default if None

    Returns:
        Number of icons in database, 0 if database doesn't exist or error
    """
    if db_path is None:
        db_path = get_default_db_path()

    if not db_path or not pathlib.Path(db_path).exists():
        return 0

    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM icons")
            result = cursor.fetchone()
            return result[0] if result else 0
    except sqlite3.Error:
        return 0
