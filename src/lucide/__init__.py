"""python-lucide - A Python package for working with Lucide icons.

This package provides utilities for creating, accessing, and using Lucide
icons from a SQLite database.
"""

__version__ = "0.1.0"

from .core import create_placeholder_svg, get_icon_list, lucide_icon
from .db import get_db_connection, get_default_db_path

__all__ = [
    "create_placeholder_svg",
    "get_db_connection",
    "get_default_db_path",
    "get_icon_list",
    "lucide_icon",
]
