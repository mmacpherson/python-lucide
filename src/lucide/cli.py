#!/usr/bin/env python3
"""Command-line interface for Lucide icon tools.

Entry points::

    lucide db            Build the icon database from the Lucide repo
    lucide describe      Generate icon descriptions via VLM (Gemini)
    lucide build-search  Build the search SQLite DB from a descriptions JSONL
    lucide search        Semantic search for icons
    lucide cluster       Discover semantic clusters in embedding space
    lucide version       Check for Lucide version updates

The legacy ``lucide-db`` command is preserved as an alias for ``lucide db``.
"""

import argparse
import dataclasses
import json
import logging
import pathlib
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime

from .config import (
    DEFAULT_LUCIDE_TAG,
    DEFAULT_SEARCH_MODEL_ID,
    EMBEDDING_MODELS,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DatabaseReportData:
    """Data class for database report information."""

    added_count: int
    skipped_count: int
    icons_to_include: set
    svg_files: list
    cursor: sqlite3.Cursor
    verbose: bool = False


# ---------------------------------------------------------------------------
# DB-build internals (unchanged)
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool) -> None:
    """Configure logging level based on verbosity flag."""
    log_level = logging.INFO if verbose else logging.ERROR
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")


def _parse_icon_filter(
    icon_list: list[str] | None, icon_file: pathlib.Path | str | None
) -> tuple[set[str] | None, bool]:
    """Prepare the set of icons to include.

    Args:
        icon_list: Optional list of specific icon names to include.
        icon_file: Optional file containing icon names (one per line).

    Returns:
        (icons_to_include set, success boolean)
    """
    icons_to_include: set[str] = set()
    if icon_list:
        icons_to_include.update(icon_list)

    if icon_file:
        try:
            with open(icon_file) as f:
                file_icons = [line.strip() for line in f if line.strip()]
                icons_to_include.update(file_icons)
            logger.info("Loaded %d icon names from %s", len(file_icons), icon_file)
        except Exception as e:
            logger.error("Failed to read icon file %s: %s", icon_file, e)
            return None, False

    return icons_to_include, True


def _clone_repository(
    temp_path: pathlib.Path, tag: str, verbose: bool
) -> tuple[pathlib.Path | None, bool]:
    """Clone the Lucide repository with the specified tag.

    Args:
        temp_path: Path to clone into.
        tag: Lucide version tag to download.
        verbose: If True, prints detailed progress information.

    Returns:
        (icons_dir path, success boolean)
    """
    repo_url = "https://github.com/lucide-icons/lucide.git"
    logger.info("Cloning tag %s of %s (shallow)...", tag, repo_url)

    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                "advice.detachedHead=false",
                "clone",
                "--quiet",
                "--depth=1",
                f"--branch={tag}",
                repo_url,
                str(temp_path / "lucide"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        if verbose:
            if result.stdout:
                logger.info("Git stdout: %s", result.stdout)
            if result.stderr:
                logger.info("Git stderr: %s", result.stderr)
    except subprocess.CalledProcessError as e:
        logger.error("Git clone failed: %s", e)
        logger.error("Output: %s", e.stderr)
        return None, False
    except FileNotFoundError:
        logger.error("Git command not found. Please install git.")
        return None, False

    icons_dir = temp_path / "lucide" / "icons"
    if not icons_dir.exists():
        logger.error("Icons directory not found: %s", icons_dir)
        return None, False

    logger.info("Found icons directory: %s", icons_dir)
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
        output_path: Path where the database will be saved.
        icons_dir: Directory containing the icon SVG files.
        icons_to_include: Set of icon names to include (or empty for all).
        tag: Lucide version tag used for this database.
        verbose: If True, prints detailed progress information.

    Returns:
        Success or failure.
    """
    if verbose:
        logger.info("Creating SQLite database: %s", output_path)

    try:
        conn = sqlite3.connect(output_path)
        cursor = conn.cursor()

        tables = ("icons", "metadata", "icon_tags", "icon_categories", "icon_aliases")
        for table in tables:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")

        cursor.execute("CREATE TABLE icons (name TEXT PRIMARY KEY, svg TEXT NOT NULL)")
        cursor.execute(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        cursor.execute(
            "CREATE TABLE icon_tags"
            " (name TEXT NOT NULL, tag TEXT NOT NULL, UNIQUE(name, tag))"
        )
        cursor.execute(
            "CREATE TABLE icon_categories"
            " (name TEXT NOT NULL, category TEXT NOT NULL, UNIQUE(name, category))"
        )
        cursor.execute(
            "CREATE TABLE icon_aliases"
            " (name TEXT NOT NULL, alias TEXT NOT NULL, UNIQUE(name, alias))"
        )

        current_time = datetime.now().isoformat()
        cursor.execute("INSERT INTO metadata VALUES (?, ?)", ("version", tag))
        cursor.execute(
            "INSERT INTO metadata VALUES (?, ?)", ("created_at", current_time)
        )

        svg_files = list(icons_dir.glob("*.svg"))
        if verbose:
            logger.info("Found %d SVG files", len(svg_files))

        added_count, skipped_count = _add_icons_to_db(
            cursor, svg_files, icons_to_include, verbose
        )

        # Populate tags, categories, and aliases from JSON sidecar files
        _add_metadata_to_db(cursor, icons_dir, icons_to_include, verbose)

        # Indexes for search
        cursor.execute("CREATE INDEX idx_tag ON icon_tags(tag)")
        cursor.execute("CREATE INDEX idx_category ON icon_categories(category)")
        cursor.execute("CREATE INDEX idx_alias ON icon_aliases(alias)")

        conn.commit()
        cursor.execute("VACUUM")

        report_data = DatabaseReportData(
            added_count=added_count,
            skipped_count=skipped_count,
            icons_to_include=icons_to_include,
            svg_files=svg_files,
            cursor=cursor,
            verbose=verbose,
        )
        _report_database_results(report_data)

        conn.close()

        if verbose:
            logger.info("Database created successfully at: %s", output_path)
        else:
            logger.debug("Database created successfully at: %s", output_path)
        return True

    except sqlite3.Error as e:
        logger.error("SQLite error: %s", e)
        return False
    except Exception as e:
        logger.error("Error building database: %s", e)
        return False


def _add_icons_to_db(
    cursor: sqlite3.Cursor,
    svg_files: list[pathlib.Path],
    icons_to_include: set[str],
    verbose: bool = False,
) -> tuple[int, int]:
    """Add icons to the database.

    Args:
        cursor: SQLite cursor.
        svg_files: List of SVG files to process.
        icons_to_include: Set of icon names to include (or empty for all).
        verbose: If True, prints detailed progress information.

    Returns:
        (added_count, skipped_count)
    """
    added_count = 0
    skipped_count = 0

    total_files = len(svg_files)
    if verbose and total_files > 0:
        logger.info("Processing %d SVG files...", total_files)

    for i, svg_file in enumerate(svg_files):
        name = svg_file.stem

        if icons_to_include and name not in icons_to_include:
            skipped_count += 1
            continue

        with open(svg_file, encoding="utf-8") as f:
            svg_content = f.read()

        cursor.execute("INSERT INTO icons VALUES (?, ?)", (name, svg_content))
        added_count += 1

        if verbose and (i + 1) % 100 == 0:
            logger.info("Processed %d/%d SVG files...", i + 1, total_files)

    return added_count, skipped_count


def _add_metadata_to_db(
    cursor: sqlite3.Cursor,
    icons_dir: pathlib.Path,
    icons_to_include: set[str],
    verbose: bool = False,
) -> None:
    """Read Lucide JSON sidecar files and populate tags/categories/aliases.

    Args:
        cursor: SQLite cursor.
        icons_dir: Directory containing icon JSON metadata files.
        icons_to_include: Set of icon names to include (or empty for all).
        verbose: If True, prints progress.
    """
    json_files = list(icons_dir.glob("*.json"))
    malformed_count = 0
    for json_file in json_files:
        name = json_file.stem
        if icons_to_include and name not in icons_to_include:
            continue
        try:
            data = json.loads(json_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping malformed metadata file %s: %s", json_file, e)
            malformed_count += 1
            continue

        for tag in data.get("tags", []):
            cursor.execute("INSERT OR IGNORE INTO icon_tags VALUES (?, ?)", (name, tag))
        for cat in data.get("categories", []):
            cursor.execute(
                "INSERT OR IGNORE INTO icon_categories VALUES (?, ?)", (name, cat)
            )
        for alias_entry in data.get("aliases", []):
            alias_name = (
                alias_entry
                if isinstance(alias_entry, str)
                else alias_entry.get("name", "")
            )
            if alias_name:
                cursor.execute(
                    "INSERT OR IGNORE INTO icon_aliases VALUES (?, ?)",
                    (name, alias_name),
                )

    if verbose:
        counts = {
            table: cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("icon_tags", "icon_categories", "icon_aliases")
        }
        logger.info(
            "Added metadata from %d JSON files: %d tags, %d categories,"
            " %d aliases (%d malformed files skipped)",
            len(json_files),
            counts["icon_tags"],
            counts["icon_categories"],
            counts["icon_aliases"],
            malformed_count,
        )


def _report_database_results(data: DatabaseReportData) -> None:
    """Report the results of the database creation.

    Args:
        data: DatabaseReportData instance containing report information.
    """
    if data.icons_to_include:
        if data.verbose:
            logger.info(
                "Added %d icons to the database, skipped %d",
                data.added_count,
                data.skipped_count,
            )

        svg_file_stems = {svg_file.stem for svg_file in data.svg_files}
        missing_icons = data.icons_to_include - svg_file_stems
        if missing_icons:
            logger.warning(
                "The following requested icons were not found: %s",
                ", ".join(missing_icons),
            )
    elif data.verbose:
        logger.info("Added all %d icons to the database", data.added_count)

    if data.verbose:
        data.cursor.execute("SELECT name FROM icons LIMIT 5")
        sample_icons = [row[0] for row in data.cursor.fetchall()]
        logger.info("Sample icons in database: %s", ", ".join(sample_icons))


# ---------------------------------------------------------------------------
# Public API for DB building
# ---------------------------------------------------------------------------


def download_and_build_db(
    output_path: pathlib.Path | str | None = None,
    tag: str = DEFAULT_LUCIDE_TAG,
    icon_list: list[str] | None = None,
    icon_file: pathlib.Path | str | None = None,
    verbose: bool = False,
) -> pathlib.Path | None:
    """Download Lucide icons and build a SQLite database.

    Args:
        output_path: Path where the database will be saved.  If None,
            saves in current directory.
        tag: Lucide version tag to download.
        icon_list: Optional list of specific icon names to include.
        icon_file: Optional file containing icon names (one per line).
        verbose: If True, prints detailed progress information.

    Returns:
        Path to the created database file or None if the operation failed.
    """
    _setup_logging(verbose)
    icons_to_include, success = _parse_icon_filter(icon_list, icon_file)
    if not success:
        return None

    if output_path is None:
        output_path = pathlib.Path.cwd() / "lucide-icons.db"
    else:
        output_path = pathlib.Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        if verbose:
            logger.info("Created temporary directory: %s", temp_path)

        icons_dir, success = _clone_repository(temp_path, tag, verbose)
        if not success or icons_dir is None:
            return None

        icons_set = set() if icons_to_include is None else icons_to_include
        success = _create_database(output_path, icons_dir, icons_set, tag, verbose)
        if not success:
            return None

        return output_path


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _cmd_db(args: argparse.Namespace) -> int:
    """Handle ``lucide db``."""
    icon_list: list[str] | None = None
    if args.icons:
        icon_list = [icon.strip() for icon in args.icons.split(",") if icon.strip()]

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


def _cmd_describe(args: argparse.Namespace) -> int:
    """Handle ``lucide describe``."""
    from .build_search import generate_descriptions  # noqa: PLC0415

    _setup_logging(args.verbose)

    icons_db = _resolve_icons_db(args.icons_db)
    if icons_db is None:
        return 1

    output = pathlib.Path(
        args.output or icons_db.parent / "gemini-icon-descriptions.jsonl"
    )
    icons_dir = pathlib.Path(args.icons_dir) if args.icons_dir else None

    try:
        count = generate_descriptions(
            icons_db,
            output,
            icons_dir=icons_dir,
            api_key=args.gemini_api_key,
            incremental=not args.no_incremental,
            verbose=args.verbose,
        )
        print(f"Generated {count} descriptions → {output}")
    except Exception as e:
        print(f"Error: {e}")
        if not args.verbose:
            print("Run with --verbose for more details.")
        return 1
    return 0


def _cmd_build_search(args: argparse.Namespace) -> int:
    """Handle ``lucide build-search``."""
    from .build_search import build_search_db  # noqa: PLC0415

    _setup_logging(args.verbose)

    descriptions_file = pathlib.Path(args.descriptions_file)
    if not descriptions_file.exists():
        print(f"Descriptions file not found: {descriptions_file}")
        print("Generate descriptions first with: lucide describe")
        return 1

    clusters_file = pathlib.Path(args.clusters_file)
    if not clusters_file.exists():
        print(f"Clusters file not found: {clusters_file}")
        print("Generate clusters first with: lucide cluster")
        return 1

    search_db = pathlib.Path(
        args.output or descriptions_file.parent / "lucide-search.db"
    )

    icons_db = pathlib.Path(args.icons_db) if args.icons_db else None

    try:
        build_search_db(
            descriptions_file,
            search_db,
            clusters_file,
            icons_db_path=icons_db,
            verbose=args.verbose,
        )
        print(f"Search DB built → {search_db}")
    except Exception as e:
        print(f"Error: {e}")
        if not args.verbose:
            print("Run with --verbose for more details.")
        return 1
    return 0


def _first_run_downloads(model: str) -> list[str]:
    """Human-readable one-time downloads the next search will trigger.

    Best-effort: peeks at fastembed's model cache and the search-DB cache;
    any surprise in their layout just means the notice is skipped.
    """
    from .search import search_available  # noqa: PLC0415

    pending = []
    try:
        from fastembed import TextEmbedding  # noqa: PLC0415
        from fastembed.common.utils import define_cache_dir  # noqa: PLC0415

        fastembed_name = EMBEDDING_MODELS[model].fastembed_model
        meta = next(
            m
            for m in TextEmbedding.list_supported_models()
            if m["model"] == fastembed_name
        )
        hf_repo = (meta.get("sources") or {}).get("hf") or ""
        size_mb = round(meta["size_in_GB"] * 1000)
        model_dir = (
            pathlib.Path(define_cache_dir(None))
            / f"models--{hf_repo.replace('/', '--')}"
        )
        if hf_repo and not model_dir.exists():
            pending.append(f"embedding model (~{size_mb} MB)")
    except Exception:
        pass
    if not search_available():
        pending.append("search index (11 MB)")
    return pending


def _cmd_search(args: argparse.Namespace) -> int:
    """Handle ``lucide search``."""
    import os  # noqa: PLC0415

    from .search import search_icons  # noqa: PLC0415

    _setup_logging(args.verbose)

    if args.search_db:
        os.environ["LUCIDE_SEARCH_DB_PATH"] = str(args.search_db)
    elif not os.environ.get("LUCIDE_SEARCH_DB_PATH"):
        # Check for search DB alongside the icons DB
        from .db import get_default_db_path  # noqa: PLC0415

        icons_db = get_default_db_path()
        if icons_db:
            candidate = icons_db.parent / "lucide-search.db"
            if candidate.exists():
                os.environ["LUCIDE_SEARCH_DB_PATH"] = str(candidate)

    # Downloading without a token is expected here; the hf_xet download
    # backend's HF_TOKEN nudge reads like an error mid-download, so keep
    # warnings from its logger out of the way
    logging.getLogger("hf_xet").setLevel(logging.ERROR)

    pending = _first_run_downloads(args.model)
    if pending:
        print(
            f"First run: downloading {' and '.join(pending)} — "
            "cached for future searches.",
            file=sys.stderr,
        )

    try:
        results = search_icons(args.query, limit=args.limit, model=args.model)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    if not results:
        print("No results found.")
        return 0

    show_icons, graphics_tip = _graphics_support()
    if graphics_tip:
        print(graphics_tip, file=sys.stderr)
    svgs: dict[str, str] = {}
    if show_icons:
        svgs = _load_svgs_for_results([r.name for r in results])

    dim = "\033[2m"
    bold = "\033[1m"
    reset = "\033[0m"
    cyan = "\033[38;2;100;200;220m"

    for r in results:
        prefix = "  "
        if show_icons:
            drawn = r.name in svgs and _display_kitty_image(svgs[r.name])
            # The image sits in columns 1-2 (cursor untouched, C=1), so
            # step over it to keep names aligned whether or not it drew
            prefix = "\033[3C" if drawn else "   "
        print(f"{prefix}{bold}{cyan}{r.name}{reset}  {dim}{r.score:.0%}{reset}")
        if args.verbose and r.description:
            print(f"    {dim}{r.description[:100]}{reset}")

    return 0


def _graphics_support() -> tuple[bool, str | None]:
    """Decide whether to render inline icons via the Kitty graphics protocol.

    Returns:
        A ``(show_icons, tip)`` pair. *tip* is a one-line stderr hint for
        the cases where the terminal could show icons but something the
        user can fix is in the way (tmux passthrough off, cairosvg not
        installed) — silently dropping the feature proved confusing.
    """
    import os  # noqa: PLC0415

    term = os.environ.get("TERM_PROGRAM", "")
    host_capable = (
        term in ("ghostty", "kitty", "WezTerm")
        or bool(os.environ.get("GHOSTTY_RESOURCES_DIR"))
        or bool(os.environ.get("KITTY_WINDOW_ID"))
        or "kitty" in os.environ.get("TERM", "")
    )
    if not host_capable:
        return False, None

    if os.environ.get("TMUX") and not _tmux_allows_passthrough():
        return False, (
            "Tip: your terminal can show icons inline, but tmux blocks "
            "them — run: tmux set -g allow-passthrough on"
        )

    try:
        import cairosvg  # noqa: F401, PLC0415
    except ImportError:
        return False, (
            "Tip: your terminal can show icons inline — install cairosvg "
            "to enable (pip install cairosvg, or uvx --with cairosvg)."
        )

    return True, None


def _tmux_allows_passthrough() -> bool:
    """Check whether the current tmux pane permits escape passthrough."""
    try:
        proc = subprocess.run(
            ["tmux", "show", "-Apv", "allow-passthrough"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return False
    return proc.stdout.strip() in ("on", "all")


def _load_svgs_for_results(names: list[str]) -> dict[str, str]:
    """Load SVG content for the given icon names from the main DB."""
    from .db import get_db_connection  # noqa: PLC0415

    svgs: dict[str, str] = {}
    with get_db_connection() as conn:
        if conn is None:
            return svgs
        for name in names:
            row = conn.execute(
                "SELECT svg FROM icons WHERE name = ?", (name,)
            ).fetchone()
            if row:
                svgs[name] = row[0]
    return svgs


def _display_kitty_image(svg_content: str) -> bool:
    """Render an SVG as a one-row inline image at the cursor position.

    Emits with ``C=1`` so the cursor stays put and the caller controls
    layout with ordinary cursor movement — which is also what makes tmux
    passthrough safe: tmux never sees the image escape, and everything
    that *does* move the cursor goes through tmux normally.

    Returns:
        True if an image escape was written.
    """
    import base64  # noqa: PLC0415
    import os  # noqa: PLC0415

    try:
        import cairosvg  # noqa: PLC0415
    except ImportError:
        return False

    try:
        # Insert a white background rect so icons show on any terminal bg
        bg_rect = '<rect width="100%" height="100%" fill="white" rx="3"/>'
        padded_svg = svg_content.replace(">", f">{bg_rect}", 1)
        # Rendered larger than displayed; the terminal downscales, which
        # keeps strokes crisp on hidpi displays
        png_data: bytes = cairosvg.svg2png(
            bytestring=padded_svg.encode("utf-8"),
            output_width=96,
            output_height=96,
        )
        b64 = base64.b64encode(png_data).decode("ascii")
        seq = f"\033_Gf=100,t=d,a=T,r=1,c=2,C=1,q=2;{b64}\033\\"
        if os.environ.get("TMUX"):
            seq = "\033Ptmux;" + seq.replace("\033", "\033\033") + "\033\\"
        sys.stdout.write(seq)
        sys.stdout.flush()
    except Exception:
        return False
    return True


def _cmd_cluster(args: argparse.Namespace) -> int:
    """Handle ``lucide cluster``."""
    from .build_clusters import (  # noqa: PLC0415
        build_cluster_visualization,
        discover_clusters,
        name_clusters,
        save_clusters_json,
    )

    _setup_logging(args.verbose)

    search_db: pathlib.Path | None
    if args.search_db:
        search_db = pathlib.Path(args.search_db)
    else:
        # Mirror `lucide search`: look for the search DB alongside the icons DB
        from .db import get_default_db_path  # noqa: PLC0415

        icons_db = get_default_db_path()
        search_db = icons_db.parent / "lucide-search.db" if icons_db else None

    if search_db is None or not search_db.exists():
        print(f"Search DB not found: {search_db}")
        print("Build it first with: lucide build-search")
        return 1

    output = pathlib.Path(args.output or search_db.parent / "lucide-icon-clusters.json")

    try:
        data = discover_clusters(search_db, min_cluster_size=args.min_cluster_size)
        data = name_clusters(data, api_key=args.gemini_api_key)
        save_clusters_json(data, output)
        print(f"Clusters saved → {output}")

        if args.html:
            html_path = pathlib.Path(args.html)
            build_cluster_visualization(data, html_path)
            print(f"Visualization → {html_path}")
    except Exception as e:
        print(f"Error: {e}")
        if not args.verbose:
            print("Run with --verbose for more details.")
        return 1
    return 0


def _cmd_version(_args: argparse.Namespace) -> int:
    """Handle ``lucide version``."""
    from .dev_utils import print_version_status  # noqa: PLC0415

    print_version_status()
    return 0


def _resolve_icons_db(
    explicit_path: str | None,
) -> pathlib.Path | None:
    """Resolve the icons database path from an argument or defaults."""
    path: pathlib.Path | None
    if explicit_path:
        path = pathlib.Path(explicit_path)
    else:
        from .db import get_default_db_path  # noqa: PLC0415

        path = get_default_db_path()

    if path is None or not path.exists():
        print(f"Icons database not found: {path}")
        print("Build it first with: lucide db")
        return None
    return path


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def main() -> int:
    """Unified ``lucide`` CLI with subcommands."""
    parser = argparse.ArgumentParser(
        prog="lucide",
        description="Lucide icon tools",
    )
    subparsers = parser.add_subparsers(dest="command")

    # -- lucide db --------------------------------------------------------
    db_parser = subparsers.add_parser(
        "db",
        help="Build the icon database from the Lucide repo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    db_parser.add_argument(
        "-o",
        "--output",
        help="Output path for the SQLite database",
        default=None,
    )
    db_parser.add_argument(
        "-t",
        "--tag",
        help="Lucide version tag to download",
        default=DEFAULT_LUCIDE_TAG,
    )
    db_parser.add_argument(
        "-i",
        "--icons",
        help="Comma-separated list of icon names to include",
        default=None,
    )
    db_parser.add_argument(
        "-f",
        "--file",
        help="Path to a file containing icon names (one per line)",
        default=None,
    )
    db_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    # -- lucide describe --------------------------------------------------
    desc_parser = subparsers.add_parser(
        "describe",
        help="Generate icon descriptions via VLM (Gemini)",
    )
    desc_parser.add_argument(
        "--icons-db",
        help="Path to the icons database (auto-detected if omitted)",
        default=None,
    )
    desc_parser.add_argument(
        "-o",
        "--output",
        help="Output JSONL path (default: alongside icons DB)",
        default=None,
    )
    desc_parser.add_argument(
        "--icons-dir",
        help="Lucide repo icons/ directory (auto-cloned if omitted)",
        default=None,
    )
    desc_parser.add_argument(
        "--gemini-api-key",
        help="Gemini API key (or set GEMINI_API_KEY env var)",
        default=None,
    )
    desc_parser.add_argument(
        "--no-incremental",
        action="store_true",
        help="Regenerate all descriptions, not just new icons",
    )
    desc_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    # -- lucide build-search ----------------------------------------------
    build_parser = subparsers.add_parser(
        "build-search",
        help="Build search SQLite DB from descriptions JSONL",
    )
    build_parser.add_argument(
        "--descriptions-file",
        help="Input JSONL file with icon descriptions",
        required=True,
    )
    build_parser.add_argument(
        "--clusters-file",
        help="Input JSON file with cluster assignments",
        required=True,
    )
    build_parser.add_argument(
        "--icons-db",
        help="Icons database for filtering and version metadata",
        default=None,
    )
    build_parser.add_argument(
        "-o",
        "--output",
        help="Output path for the search database",
        default=None,
    )
    build_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    # -- lucide search ----------------------------------------------------
    search_parser = subparsers.add_parser(
        "search",
        help="Semantic search for icons",
    )
    search_parser.add_argument(
        "query",
        help="Natural language search query",
    )
    search_parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=10,
        help="Number of results to show",
    )
    search_parser.add_argument(
        "--search-db",
        help="Path to the search database",
        default=None,
    )
    search_parser.add_argument(
        "--model",
        choices=sorted(EMBEDDING_MODELS),
        default=DEFAULT_SEARCH_MODEL_ID,
        help="Embedding model to search with",
    )
    search_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show descriptions in results",
    )

    # -- lucide cluster ---------------------------------------------------
    cluster_parser = subparsers.add_parser(
        "cluster",
        help="Discover and name semantic clusters in the embedding space",
    )
    cluster_parser.add_argument(
        "--search-db",
        help="Path to the search database",
        default=None,
    )
    cluster_parser.add_argument(
        "-o",
        "--output",
        help="Output JSON path for cluster data",
        default=None,
    )
    cluster_parser.add_argument(
        "--html",
        help="Also generate an interactive HTML visualization",
        default=None,
    )
    cluster_parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="Minimum icons to form a cluster",
    )
    cluster_parser.add_argument(
        "--gemini-api-key",
        help="Gemini API key for cluster naming (or set GEMINI_API_KEY)",
        default=None,
    )
    cluster_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    # -- lucide version ---------------------------------------------------
    subparsers.add_parser(
        "version",
        help="Check for Lucide version updates",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    handlers = {
        "db": _cmd_db,
        "describe": _cmd_describe,
        "build-search": _cmd_build_search,
        "search": _cmd_search,
        "cluster": _cmd_cluster,
        "version": _cmd_version,
    }
    return handlers[args.command](args)


def main_legacy_db() -> int:
    """Legacy ``lucide-db`` entry point — delegates to ``lucide db``."""
    sys.argv = ["lucide", "db", *sys.argv[1:]]
    return main()


if __name__ == "__main__":
    sys.exit(main())
