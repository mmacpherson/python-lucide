"""Tests for cluster naming and the shipped cluster data."""

import json
import pathlib

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from lucide.build_clusters import MAX_THEME_LENGTH, _sanitize_theme, name_clusters

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


def _cluster_data() -> dict:
    return {
        "clusters": {
            "-1": {"icons": ["stray-icon"], "theme": None},
            "3": {"icons": ["dice-1", "dice-2", "dices"], "theme": None},
        }
    }


class TestNameClusters:
    def test_populates_themes(self):
        named = name_clusters(_cluster_data(), model=TestModel())

        assert named["clusters"]["-1"]["theme"] == "Unclustered"
        theme = named["clusters"]["3"]["theme"]
        assert theme
        assert "\n" not in theme
        assert len(theme) <= MAX_THEME_LENGTH

    def test_leaked_reasoning_triggers_retry(self):
        calls = 0

        def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            theme = "THINK\nThe user wants..." if calls == 1 else "Gaming Dice"
            tool_name = info.output_tools[0].name
            return ModelResponse(parts=[ToolCallPart(tool_name, {"theme": theme})])

        named = name_clusters(_cluster_data(), model=FunctionModel(respond))

        assert calls == 2
        assert named["clusters"]["3"]["theme"] == "Gaming Dice"

    def test_falls_back_after_exhausted_retries(self):
        def always_leak(
            _messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            tool_name = info.output_tools[0].name
            return ModelResponse(
                parts=[ToolCallPart(tool_name, {"theme": "THINK\nreasoning..."})]
            )

        named = name_clusters(_cluster_data(), model=FunctionModel(always_leak))

        assert named["clusters"]["3"]["theme"] == "Cluster 3"

    def test_plain_text_response_is_rejected_not_stored(self):
        # A model answering in prose instead of calling the output tool must
        # not end up as a theme
        def prose(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart("Here are my thoughts on a name...")])

        named = name_clusters(_cluster_data(), model=FunctionModel(prose))

        assert named["clusters"]["3"]["theme"] == "Cluster 3"


class TestShippedClusterData:
    def test_all_themes_are_plausible_names(self):
        data = json.loads(CLUSTERS_JSON.read_text())
        offenders = {
            lid: cluster["theme"]
            for lid, cluster in data["clusters"].items()
            if "\n" in cluster["theme"] or len(cluster["theme"]) > MAX_THEME_LENGTH
        }
        assert not offenders, f"Clusters with leaked/invalid themes: {offenders}"
