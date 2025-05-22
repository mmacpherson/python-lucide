#!/usr/bin/env python3
"""Serves lucide SVG icons from a SQLite database.

This module provides functions to retrieve Lucide icons from a SQLite database.
"""

import functools
import logging
import sqlite3
import xml.etree.ElementTree as ET

from .db import get_db_connection, get_default_db_path

logger = logging.getLogger(__name__)

# Get the default database path
# This DB_PATH is primarily for informational purposes or if other parts of core
# needed it directly.
# The functions lucide_icon and get_icon_list use get_db_connection(),
# which itself calls get_default_db_path() if no specific path is provided.
DB_PATH = get_default_db_path()


def _process_classes(root, cls, param_attrs):
    """Process CSS classes for the SVG element.

    Args:
        root: ET root element of the SVG
        cls: Optional CSS class string to append
        param_attrs: Dictionary of attributes, may contain 'class'

    Returns:
        Updated attributes dictionary with processed classes
    """
    final_attributes = root.attrib.copy()
    working_class_list = []

    # 1. Determine the base list of classes
    if "class" in param_attrs:
        # Class from param_attrs takes precedence
        class_str_from_attrs = param_attrs.pop("class", "")
        if class_str_from_attrs and isinstance(class_str_from_attrs, str):
            working_class_list.extend(c for c in class_str_from_attrs.split() if c)
    elif "class" in final_attributes:
        # Use original SVG classes if no override
        original_class_str = final_attributes.get("class", "")
        if original_class_str and isinstance(original_class_str, str):
            working_class_list.extend(c for c in original_class_str.split() if c)

    # 2. Add classes from the 'cls' parameter
    if cls and isinstance(cls, str):
        for c_item in cls.split():
            if c_item and c_item not in working_class_list:
                working_class_list.append(c_item)

    # 3. Update final_attributes with the new class string
    if working_class_list:
        final_attributes["class"] = " ".join(working_class_list)
    elif "class" in final_attributes:
        del final_attributes["class"]

    return final_attributes


def _apply_attributes(root, final_attributes, param_attrs):
    """Apply attributes to the SVG element.

    Args:
        root: ET root element of the SVG
        final_attributes: Dictionary of processed attributes
        param_attrs: Dictionary of attributes to apply

    Returns:
        None (modifies root in place)
    """
    # Apply other attributes from param_attrs (excluding 'class' which was handled)
    for key, value in param_attrs.items():
        final_attributes[key] = str(value)  # Ensure value is string for ET

    # Apply all attributes to the root element
    root.attrib.clear()
    for key, value in final_attributes.items():
        root.set(key, value)


def _modify_svg(original_svg_content, icon_name, cls, attrs):
    """Modifies the SVG content with provided attributes and classes.

    Args:
        original_svg_content: Original SVG string
        icon_name: Name of the icon (for error reporting)
        cls: Optional CSS classes to add
        attrs: Optional attributes to apply

    Returns:
        Modified SVG content as string
    """
    try:
        root = ET.fromstring(original_svg_content)
        param_attrs = dict(attrs) if attrs else {}

        # Process classes and get updated attributes
        final_attributes = _process_classes(root, cls, param_attrs)

        # Apply all attributes to the SVG
        _apply_attributes(root, final_attributes, param_attrs)

        # Serialize back to string
        return ET.tostring(root, encoding="unicode", xml_declaration=False)

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


@functools.lru_cache(maxsize=128)
def lucide_icon(
    icon_name: str, cls: str = "", attrs=None, fallback_text: str | None = None
):
    """Fetches a Lucide icon SVG from the database with caching.

    Args:
        icon_name: Name of the Lucide icon to fetch.
        cls: Optional CSS class string to apply/append to the SVG element.
             Multiple classes can be space-separated.
        attrs: Optional dictionary or frozenset of tuples of attributes to apply to
               the SVG element. If 'class' is a key in attrs, its value will determine
               the base
               classes before classes from the `cls` param are appended.
               Other attributes
               in `attrs` will override existing attributes on the SVG.
        fallback_text: Optional text to display if the icon is not found.

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

            # If no modifications needed, return the original SVG
            if not cls and not attrs:
                return original_svg_content

            # Otherwise, modify the SVG with the provided attributes and classes
            return _modify_svg(original_svg_content, icon_name, cls, attrs)

    except sqlite3.Error as e:
        logger.error(f"Database query error for icon '{icon_name}': {e}")
        return create_placeholder_svg(icon_name, fallback_text, f"DB Error: {e}")
    except Exception as e:  # Catch-all for unexpected issues
        logger.error(
            f"An unexpected error occurred while fetching icon '{icon_name}': {e}"
        )
        return create_placeholder_svg(icon_name, fallback_text, f"Error: {e}")


def create_placeholder_svg(icon_name, fallback_text=None, error_text=None):
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
stroke-linejoin="round" class="lucide lucide-placeholder"
data-missing-icon="{icon_name}">
  <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
  <text x="12" y="14" text-anchor="middle" font-size="8"
  font-family="sans-serif" fill="currentColor">{display_text}</text>
</svg>""".strip()


def get_icon_list():
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
