"""Materialize the icon database during wheel and editable builds."""

import subprocess
import sys
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class IconDatabaseBuildHook(BuildHookInterface):
    """Ship SQLite in wheels while keeping only SQL in source distributions."""

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Validate and include a freshly reconstructed database."""
        if self.target_name != "wheel":
            return
        root = Path(self.root)
        subprocess.run(
            [sys.executable, str(root / "build_support.py"), "restore"], check=True
        )
        if version != "editable":
            build_data["force_include"][
                str(root / "src/lucide/data/lucide-icons.db")
            ] = "lucide/data/lucide-icons.db"
