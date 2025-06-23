#!/usr/bin/env python3
"""Command-line interface for downloading and building the Lucide icons database."""

import argparse
import dataclasses
import logging
import pathlib
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime

from .config import DEFAULT_LUCIDE_TAG


@dataclasses.dataclass
class DatabaseReportData:
    """Data class for database report information."""

    added_count: int
    skipped_count: int
    icons_to_include: set
    svg_files: list
    cursor: sqlite3.Cursor
    verbose: bool = False


logger = logging.getLogger(__name__)


def _setup_logging_and_icons(
    verbose: bool, icon_list: list[str] | None, icon_file: pathlib.Path | str | None
) -> tuple[set[str] | None, bool]:
    """Set up logging and prepare the list of icons to include.

    Args:
        verbose: If True, prints detailed progress information
        icon_list: Optional list of specific icon names to include
        icon_file: Optional file containing icon names (one per line)

    Returns:
        tuple: (icons_to_include set, success boolean)
    """
    # Set up logging - use ERROR as default to minimize output unless verbose
    log_level = logging.INFO if verbose else logging.ERROR
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    # Get the list of icons to include
    icons_to_include = set()
    if icon_list:
        icons_to_include.update(icon_list)

    if icon_file:
        try:
            with open(icon_file) as f:
                file_icons = [line.strip() for line in f if line.strip()]
                icons_to_include.update(file_icons)
            logger.info(f"Loaded {len(file_icons)} icon names from {icon_file}")
        except Exception as e:
            logger.error(f"Failed to read icon file {icon_file}: {e}")
            return None, False

    return icons_to_include, True


def _clone_repository(
    temp_path: pathlib.Path, tag: str, verbose: bool
) -> tuple[pathlib.Path | None, bool]:
    """Clone the Lucide repository with the specified tag.

    Args:
        temp_path: Path to clone into
        tag: Lucide version tag to download
        verbose: If True, prints detailed progress information

    Returns:
        tuple: (icons_dir path, success boolean)
    """
    repo_url = "https://github.com/lucide-icons/lucide.git"
    logger.info(f"Cloning tag {tag} of {repo_url} (shallow)...")

    try:
        # Always capture output and use git options to reduce noise
        result = subprocess.run(
            [
                "git",
                "-c",
                "advice.detachedHead=false",  # Suppress detached HEAD advice
                "clone",
                "--quiet",  # Reduce git output verbosity
                "--depth=1",
                f"--branch={tag}",
                repo_url,
                str(temp_path / "lucide"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        # Only show git output in verbose mode
        if verbose:
            if result.stdout:
                logger.info(f"Git stdout: {result.stdout}")
            if result.stderr:
                logger.info(f"Git stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Git clone failed: {e}")
        logger.error(f"Output: {e.stderr}")
        return None, False
    except FileNotFoundError:
        logger.error("Git command not found. Please install git.")
        return None, False

    # Find the icons directory
    icons_dir = temp_path / "lucide" / "icons"
    if not icons_dir.exists():
        logger.error(f"Icons directory not found: {icons_dir}")
        return None, False

    logger.info(f"Found icons directory: {icons_dir}")
    return icons_dir, True


def _create_database(
    output_path: pathlib.Path | str,
    icons_dir: pathlib.Path,
    icons_to_include: set[str],
    tag: str = DEFAULT_LUCIDE_TAG,
    verbose: bool = False,
) -> bool:
    """Create the SQLite database with icons.

    Args:
        output_path: Path where the database will be saved
        icons_dir: Directory containing the icon SVG files
        icons_to_include: Set of icon names to include (or empty for all)
        tag: Lucide version tag used for this database
        verbose: If True, prints detailed progress information

    Returns:
        boolean: Success or failure
    """
    if verbose:
        logger.info(f"Creating SQLite database: {output_path}")

    try:
        conn = sqlite3.connect(output_path)
        cursor = conn.cursor()

        # Create tables with name as PRIMARY KEY
        cursor.execute("DROP TABLE IF EXISTS icons")
        cursor.execute("DROP TABLE IF EXISTS metadata")
        cursor.execute(
            """
            CREATE TABLE icons (
                name TEXT PRIMARY KEY,
                svg TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

        # Store metadata about this database build
        current_time = datetime.now().isoformat()
        cursor.execute("INSERT INTO metadata VALUES (?, ?)", ("version", tag))
        cursor.execute(
            "INSERT INTO metadata VALUES (?, ?)", ("created_at", current_time)
        )

        # Find all SVG files and add them to the database
        svg_files = list(icons_dir.glob("*.svg"))
        if verbose:
            logger.info(f"Found {len(svg_files)} SVG files")

        # Add icons to the database
        added_count, skipped_count = _add_icons_to_db(
            cursor, svg_files, icons_to_include, verbose
        )

        conn.commit()

        # Run VACUUM after commit to optimize database
        cursor.execute("VACUUM")

        # Report on the results
        report_data = DatabaseReportData(
            added_count=added_count,
            skipped_count=skipped_count,
            icons_to_include=icons_to_include,
            svg_files=svg_files,
            cursor=cursor,
            verbose=verbose,
        )
        _report_database_results(report_data)

        # Close the database connection
        conn.close()

        # Always log success, but at different levels depending on verbosity
        if verbose:
            logger.info(f"Database created successfully at: {output_path}")
        else:
            logger.debug(f"Database created successfully at: {output_path}")
        return True

    except sqlite3.Error as e:
        logger.error(f"SQLite error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error building database: {e}")
        return False


def _add_icons_to_db(
    cursor: sqlite3.Cursor,
    svg_files: list[pathlib.Path],
    icons_to_include: set[str],
    verbose: bool = False,
) -> tuple[int, int]:
    """Add icons to the database.

    Args:
        cursor: SQLite cursor
        svg_files: List of SVG files to process
        icons_to_include: Set of icon names to include (or empty for all)
        verbose: If True, prints detailed progress information

    Returns:
        tuple: (added_count, skipped_count)
    """
    added_count = 0
    skipped_count = 0

    total_files = len(svg_files)
    if verbose and total_files > 0:
        logger.info(f"Processing {total_files} SVG files...")

    for i, svg_file in enumerate(svg_files):
        name = svg_file.stem  # Get the filename without extension

        # Skip if not in the include list (if a list was provided)
        if icons_to_include and name not in icons_to_include:
            skipped_count += 1
            continue

        with open(svg_file, encoding="utf-8") as f:
            svg_content = f.read()

        cursor.execute("INSERT INTO icons VALUES (?, ?)", (name, svg_content))
        added_count += 1

        # Log progress every 100 icons if verbose
        if verbose and (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{total_files} SVG files...")

    return added_count, skipped_count


def _report_database_results(data: DatabaseReportData) -> None:
    """Report the results of the database creation.

    Args:
        data: DatabaseReportData instance containing report information
    """
    if data.icons_to_include:
        if data.verbose:
            logger.info(
                f"Added {data.added_count} icons to the database, "
                f"skipped {data.skipped_count}"
            )

        # Check for any requested icons that weren't found - always show warnings
        svg_file_stems = {svg_file.stem for svg_file in data.svg_files}
        missing_icons = data.icons_to_include - svg_file_stems
        if missing_icons:
            logger.warning(
                "The following requested icons were not found:"
                f" {', '.join(missing_icons)}"
            )
    elif data.verbose:
        logger.info(f"Added all {data.added_count} icons to the database")

    # Only show sample icons in verbose mode
    if data.verbose:
        data.cursor.execute("SELECT name FROM icons LIMIT 5")
        sample_icons = [row[0] for row in data.cursor.fetchall()]
        logger.info(f"Sample icons in database: {', '.join(sample_icons)}")


def download_and_build_db(
    output_path: pathlib.Path | str | None = None,
    tag: str = DEFAULT_LUCIDE_TAG,
    icon_list: list[str] | None = None,
    icon_file: pathlib.Path | str | None = None,
    verbose: bool = False,
) -> pathlib.Path | None:
    """Downloads Lucide icons and builds a SQLite database.

    Args:
        output_path: Path where the database will be saved. If None,
                     saves in current directory.
        tag: Lucide version tag to download
        icon_list: Optional list of specific icon names to include
        icon_file: Optional file containing icon names (one per line)
        verbose: If True, prints detailed progress information

    Returns:
        Path to the created database file or None if the operation failed
    """
    # Set up logging and prepare icons list
    icons_to_include, success = _setup_logging_and_icons(verbose, icon_list, icon_file)
    if not success:
        return None

    # Set output path
    if output_path is None:
        output_path = pathlib.Path.cwd() / "lucide-icons.db"
    else:
        output_path = pathlib.Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a temporary directory that will be automatically cleaned up
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        if verbose:
            logger.info(f"Created temporary directory: {temp_path}")

        # Clone the repository
        icons_dir, success = _clone_repository(temp_path, tag, verbose)
        if not success:
            return None

        # Create the database
        if icons_dir is None:
            return None

        icons_set = set() if icons_to_include is None else icons_to_include
        success = _create_database(output_path, icons_dir, icons_set, tag, verbose)
        if not success:
            return None

        return output_path


def main() -> int:
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Download Lucide icons and build a SQLite database",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output path for the SQLite database (default: ./lucide-icons.db)",
        default=None,
    )
    parser.add_argument(
        "-t",
        "--tag",
        help=f"Lucide version tag to download (default: {DEFAULT_LUCIDE_TAG})",
        default=DEFAULT_LUCIDE_TAG,
    )
    parser.add_argument(
        "-i",
        "--icons",
        help="Comma-separated list of icon names to include",
        default=None,
    )
    parser.add_argument(
        "-f",
        "--file",
        help="Path to a file containing icon names (one per line)",
        default=None,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Enable verbose output (default: quiet)",
        action="store_true",
    )

    args = parser.parse_args()

    # Process the icons list if provided
    icon_list: list[str] | None = None
    if args.icons:
        icon_list = [icon.strip() for icon in args.icons.split(",") if icon.strip()]

    # Call the function to download and build the database
    output_path = download_and_build_db(
        output_path=args.output,
        tag=args.tag,
        icon_list=icon_list,
        icon_file=args.file,
        verbose=args.verbose,
    )

    if output_path:
        print(f"Success! Database created at: {output_path}")
        return 0
    print("Failed to create database.")
    if not args.verbose:
        print("Run with --verbose for more details.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
