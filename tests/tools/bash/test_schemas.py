"""Tests for bash tool parameter schemas and constants."""

from engine.tools.builtin._bash.schemas import (
    BASH_TOOL_SCHEMA,
    DEFAULT_TIMEOUT_MS,
    KILL_GRACE_PERIOD_MS,
    MAX_OUTPUT_BYTES,
    MAX_OUTPUT_LINES,
    MAX_OUTPUT_SLIDING_WINDOW,
    PROCESS_TOOL_SCHEMA,
    YIELD_THRESHOLD_MS,
)


class TestBashToolSchema:
    def test_schema_type_is_object(self):
        assert BASH_TOOL_SCHEMA["type"] == "object"

    def test_has_command_property(self):
        assert "command" in BASH_TOOL_SCHEMA["properties"]
        assert BASH_TOOL_SCHEMA["properties"]["command"]["type"] == "string"

    def test_has_description_property(self):
        assert "description" in BASH_TOOL_SCHEMA["properties"]
        assert BASH_TOOL_SCHEMA["properties"]["description"]["type"] == "string"

    def test_has_timeout_property(self):
        assert "timeout" in BASH_TOOL_SCHEMA["properties"]
        assert BASH_TOOL_SCHEMA["properties"]["timeout"]["type"] == "number"

    def test_has_workdir_property(self):
        assert "workdir" in BASH_TOOL_SCHEMA["properties"]
        assert BASH_TOOL_SCHEMA["properties"]["workdir"]["type"] == "string"

    def test_has_env_property(self):
        assert "env" in BASH_TOOL_SCHEMA["properties"]
        assert BASH_TOOL_SCHEMA["properties"]["env"]["type"] == "object"
        assert BASH_TOOL_SCHEMA["properties"]["env"]["additionalProperties"] == {"type": "string"}

    def test_has_background_property(self):
        assert "background" in BASH_TOOL_SCHEMA["properties"]
        assert BASH_TOOL_SCHEMA["properties"]["background"]["type"] == "boolean"

    def test_required_fields(self):
        assert "command" in BASH_TOOL_SCHEMA["required"]
        assert "description" in BASH_TOOL_SCHEMA["required"]
        assert len(BASH_TOOL_SCHEMA["required"]) == 2


class TestProcessToolSchema:
    def test_schema_type_is_object(self):
        assert PROCESS_TOOL_SCHEMA["type"] == "object"

    def test_has_action_property(self):
        assert "action" in PROCESS_TOOL_SCHEMA["properties"]
        assert PROCESS_TOOL_SCHEMA["properties"]["action"]["type"] == "string"
        assert PROCESS_TOOL_SCHEMA["properties"]["action"]["enum"] == ["list", "poll", "log", "kill"]

    def test_has_session_id_property(self):
        assert "session_id" in PROCESS_TOOL_SCHEMA["properties"]
        assert PROCESS_TOOL_SCHEMA["properties"]["session_id"]["type"] == "string"

    def test_required_fields(self):
        assert "action" in PROCESS_TOOL_SCHEMA["required"]
        assert len(PROCESS_TOOL_SCHEMA["required"]) == 1


class TestConstants:
    def test_default_timeout_ms(self):
        assert DEFAULT_TIMEOUT_MS == 120_000

    def test_max_output_lines(self):
        assert MAX_OUTPUT_LINES == 2_000

    def test_max_output_bytes(self):
        assert MAX_OUTPUT_BYTES == 50 * 1024

    def test_max_output_sliding_window(self):
        assert MAX_OUTPUT_SLIDING_WINDOW == MAX_OUTPUT_BYTES * 2

    def test_yield_threshold_ms(self):
        assert YIELD_THRESHOLD_MS == 10_000

    def test_kill_grace_period_ms(self):
        assert KILL_GRACE_PERIOD_MS == 3_000

    def test_all_constants_are_ints(self):
        for val in [DEFAULT_TIMEOUT_MS, MAX_OUTPUT_LINES, MAX_OUTPUT_BYTES,
                    MAX_OUTPUT_SLIDING_WINDOW, YIELD_THRESHOLD_MS, KILL_GRACE_PERIOD_MS]:
            assert isinstance(val, int)
