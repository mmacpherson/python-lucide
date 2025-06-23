#!/usr/bin/env python3
# ruff: noqa: PLR0913
"""Serves lucide SVG icons from a SQLite database.

This module provides functions to retrieve Lucide icons from a SQLite database.
"""

import functools
import logging
import sqlite3
import xml.etree.ElementTree as ET

from .config import DEFAULT_ICON_CACHE_SIZE
from .db import get_db_connection, get_default_db_path

logger = logging.getLogger(__name__)

# Get the default database path
# This DB_PATH is primarily for informational purposes or if other parts of core
# needed it directly.
# The functions lucide_icon and get_icon_list use get_db_connection(),
# which itself calls get_default_db_path() if no specific path is provided.
DB_PATH = get_default_db_path()

SVG_NAMESPACE_URI = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NAMESPACE_URI)


def _process_classes(root: ET.Element, cls: str | None, icon_name: str) -> None:
    """Process CSS classes for the SVG element, modifying it in place.

    Args:
        root: ET.Element root element of the SVG.
        cls: Optional CSS class string to append.
        icon_name: Name of the icon to generate automatic Lucide classes.
    """
    # Start with existing classes on the SVG element
    original_class_str = root.attrib.get("class", "")
    working_classes = (
        {c for c in original_class_str.split() if c} if original_class_str else set()
    )

    # Add automatic Lucide classes
    working_classes.update(
        {"lucide", f"lucide-{icon_name}", f"lucide-{icon_name}-icon"}
    )

    # Add classes from the 'cls' parameter
    if cls and isinstance(cls, str):
        working_classes.update(cls.split())

    # Update the root element's class attribute
    if working_classes:
        root.set("class", " ".join(sorted(working_classes)))
    elif "class" in root.attrib:
        # Remove class attribute if it's empty and was present
        del root.attrib["class"]


def _apply_attributes(  # noqa: D417
    root: ET.Element,
    width: str | int | None = None,
    height: str | int | None = None,
    fill: str | None = None,
    stroke: str | None = None,
    stroke_width: str | int | None = None,
    stroke_linecap: str | None = None,
    stroke_linejoin: str | None = None,
) -> None:
    """Apply explicit attributes to the SVG element, modifying it in place.

    Args:
        root: ET.Element root element of the SVG.
        width, height, fill, stroke, stroke_width, stroke_linecap, stroke_linejoin:
            Optional SVG attributes to set or override.
            Pythonic names like 'stroke_width' are converted to 'stroke-width'.
    """
    attrs_to_set = {
        "width": width,
        "height": height,
        "fill": fill,
        "stroke": stroke,
        "stroke_width": stroke_width,
        "stroke_linecap": stroke_linecap,
        "stroke_linejoin": stroke_linejoin,
    }

    attr_name_map = {
        "stroke_width": "stroke-width",
        "stroke_linecap": "stroke-linecap",
        "stroke_linejoin": "stroke-linejoin",
    }

    for py_name, value in attrs_to_set.items():
        if value is not None:
            svg_attr_name = attr_name_map.get(py_name, py_name)
            root.set(svg_attr_name, str(value))


def _modify_svg(  # noqa: D417
    original_svg_content: str,
    icon_name: str,
    cls: str | None,
    width: int | str | None = None,
    height: int | str | None = None,
    fill: str | None = None,
    stroke: str | None = None,
    stroke_width: int | str | None = None,
    stroke_linecap: str | None = None,
    stroke_linejoin: str | None = None,
) -> str:
    """Modifies the SVG content with provided attributes and classes.

    Args:
        original_svg_content: Original SVG string.
        icon_name: Name of the icon (for error reporting).
        cls: Optional CSS classes to add/append.
        width, height, fill, stroke, stroke_width, stroke_linecap, stroke_linejoin:
            Optional SVG attributes to set or override.

    Returns:
        Modified SVG content as string.
    """
    try:
        root = ET.fromstring(original_svg_content)

        # Process and apply CSS classes
        _process_classes(root, cls, icon_name)

        # Apply explicit attributes
        _apply_attributes(
            root,
            width=width,
            height=height,
            fill=fill,
            stroke=stroke,
            stroke_width=stroke_width,
            stroke_linecap=stroke_linecap,
            stroke_linejoin=stroke_linejoin,
        )

        # Serialize back to string
        return ET.tostring(
            root, encoding="unicode", xml_declaration=False, method="xml"
        )

    except ET.ParseError as e:
        logger.warning(
            f"Failed to parse SVG for icon '{icon_name}': {e}. Using original SVG."
        )
        return original_svg_content
    except Exception as e:
        logger.warning(
            f"Failed to modify SVG for icon '{icon_name}': {e}. Using original SVG."
        )
        return original_svg_content


@functools.lru_cache(maxsize=DEFAULT_ICON_CACHE_SIZE)
def lucide_icon(
    icon_name: str,
    cls: str = "",
    fallback_text: str | None = None,
    width: str | int | None = None,
    height: str | int | None = None,
    fill: str | None = None,
    stroke: str | None = None,
    stroke_width: str | int | None = None,
    stroke_linecap: str | None = None,
    stroke_linejoin: str | None = None,
) -> str:
    """Fetches a Lucide icon SVG from the database with caching.

    Args:
        icon_name: Name of the Lucide icon to fetch.
        cls: Optional CSS class string to apply/append to the SVG element.
             Multiple classes can be space-separated.
        fallback_text: Optional text to display if the icon is not found.
        width: Optional width attribute for the SVG element.
        height: Optional height attribute for the SVG element.
        fill: Optional fill attribute for the SVG element.
        stroke: Optional stroke attribute for the SVG element.
        stroke_width: Optional stroke-width attribute for the SVG element.
        stroke_linecap: Optional stroke-linecap attribute for the SVG element.
        stroke_linejoin: Optional stroke-linejoin attribute for the SVG element.

    Returns:
        The SVG content as a string.
    """
    try:
        with get_db_connection() as conn:
            if conn is None:
                logger.error(f"Failed to connect to database for icon '{icon_name}'.")
                return create_placeholder_svg(icon_name, fallback_text)

            # Query the database
            cursor = conn.cursor()
            cursor.execute("SELECT svg FROM icons WHERE name = ?", (icon_name,))
            row = cursor.fetchone()

            if not row or not row[0]:
                logger.warning(f"Lucide icon '{icon_name}' not found in database.")
                return create_placeholder_svg(icon_name, fallback_text)

            original_svg_content = row[0]

            return _modify_svg(
                original_svg_content,
                icon_name,
                cls,
                width=width,
                height=height,
                fill=fill,
                stroke=stroke,
                stroke_width=stroke_width,
                stroke_linecap=stroke_linecap,
                stroke_linejoin=stroke_linejoin,
            )

    except sqlite3.Error as e:
        logger.error(f"Database query error for icon '{icon_name}': {e}")
        return create_placeholder_svg(icon_name, fallback_text, f"DB Error: {e}")
    except Exception as e:  # Catch-all for unexpected issues
        logger.error(
            f"An unexpected error occurred while fetching icon '{icon_name}': {e}"
        )
        return create_placeholder_svg(icon_name, fallback_text, f"Error: {e}")


def create_placeholder_svg(
    icon_name: str,
    fallback_text: str | None = None,
    error_text: str | None = None,
) -> str:
    """Creates a placeholder SVG when an icon is not found or an error occurs.

    Args:
        icon_name: The name of the requested icon.
        fallback_text: Optional text to display in the placeholder.
        error_text: Optional error message to include as a comment.

    Returns:
        A string containing an SVG placeholder.
    """
    display_text = fallback_text if fallback_text is not None else icon_name
    comment = (
        f"<!-- {error_text} -->"
        if error_text
        else f"<!-- Icon '{icon_name}' not found -->"
    )

    # Using a more robust SVG structure for the placeholder
    # Ensures it's also parsable by ElementTree for consistency in testing if needed
    return f"""{comment}
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"
fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
stroke-linejoin="round" class="lucide lucide-{icon_name} lucide-placeholder"
data-missing-icon="{icon_name}">
  <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
  <text x="12" y="14" text-anchor="middle" font-size="8"
  font-family="sans-serif" fill="currentColor">{display_text}</text>
</svg>""".strip()


def get_icon_list() -> list[str]:
    """Returns a list of all available icon names from the database.

    Returns:
        list: A list of icon names, or an empty list if the database cannot be accessed.
    """
    try:
        with get_db_connection() as conn:
            if conn is None:
                return []

            cursor = conn.cursor()
            cursor.execute("SELECT name FROM icons ORDER BY name")
            return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Error retrieving icon list: {e}")
        return []
