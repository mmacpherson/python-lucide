"""Tests for cluster naming and the shipped cluster data."""

import json
import pathlib

from lucide.build_clusters import MAX_THEME_LENGTH, _sanitize_theme

CLUSTERS_JSON = (
    pathlib.Path(__file__).parent.parent
    / "src"
    / "lucide"
    / "data"
    / "lucide-icon-clusters.json"
)


class TestSanitizeTheme:
    def test_accepts_short_name(self):
        assert _sanitize_theme("Gaming Dice") == "Gaming Dice"

    def test_strips_wrapping_quotes(self):
        assert _sanitize_theme('"Call Event Indicators"') == "Call Event Indicators"

    def test_rejects_multiline_reasoning(self):
        leaked = (
            'THINK\nThe user wants a short theme name...\nLet\'s go with "Gaming Dice".'
        )
        assert _sanitize_theme(leaked) is None

    def test_rejects_overlong_response(self):
        assert _sanitize_theme("x" * (MAX_THEME_LENGTH + 1)) is None

    def test_rejects_empty(self):
        assert _sanitize_theme('""') is None


class TestShippedClusterData:
    def test_all_themes_are_plausible_names(self):
        data = json.loads(CLUSTERS_JSON.read_text())
        offenders = {
            lid: cluster["theme"]
            for lid, cluster in data["clusters"].items()
            if "\n" in cluster["theme"] or len(cluster["theme"]) > MAX_THEME_LENGTH
        }
        assert not offenders, f"Clusters with leaked/invalid themes: {offenders}"
