"""Characterization tests for prompt assembly (XML format).

Asserts XML-tagged sections so regressions in prompt structure
are caught early.
"""

import pytest

from engine.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    build_subagent_prompt,
    build_system_prompt,
    get_subagent_system_prompt,
)
from engine.runner import _ENV_BLOCK_PATTERN


# ---------------------------------------------------------------------------
# build_system_prompt()
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:

    def test_build_system_prompt_full(self) -> None:
        result = build_system_prompt(
            include_spawn=True,
            env_context={"Date": "Mon Jan 01 2026", "OS": "Darwin"},
            tool_descriptions=[("read", "Read files"), ("grep", "Search content")],
            user_instructions="Be concise.",
        )

        assert "\n\n" in result
        assert "<core-rules>" in result
        assert "</core-rules>" in result
        assert "<spawning-guideline>" in result
        assert "</spawning-guideline>" in result
        assert "<environment>" in result
        assert "</environment>" in result
        assert "<available-tools>" in result
        assert "<user-instructions>" in result

        assert "## Environment" not in result
        assert "## Available Tools" not in result
        assert "## Custom Instructions" not in result

        pos_env = result.index("<environment>")
        pos_tools = result.index("<available-tools>")
        pos_custom = result.index("<user-instructions>")
        assert pos_env < pos_tools < pos_custom

        assert "- **Date**: Mon Jan 01 2026" in result
        assert "- **OS**: Darwin" in result
        assert "- **read**: Read files" in result
        assert "- **grep**: Search content" in result
        assert "Be concise." in result

    def test_build_system_prompt_no_spawn(self) -> None:
        with_spawn = build_system_prompt(
            include_spawn=True,
            env_context={"Date": "Mon Jan 01 2026"},
        )
        without_spawn = build_system_prompt(
            include_spawn=False,
            env_context={"Date": "Mon Jan 01 2026"},
        )

        assert len(without_spawn) < len(with_spawn)
        assert "<spawning-guideline>" not in without_spawn
        assert "<core-rules>" in without_spawn

    def test_build_system_prompt_minimal(self) -> None:
        result = build_system_prompt()

        assert "<environment>" not in result
        assert "<available-tools>" not in result
        assert "<user-instructions>" not in result
        assert "<core-rules>" in result

    def test_internal_headings_preserved_inside_xml(self) -> None:
        result = build_system_prompt(include_spawn=True)

        core_start = result.index("<core-rules>")
        core_end = result.index("</core-rules>")
        core_content = result[core_start:core_end]
        assert "# Execution Strategy" in core_content
        assert "# Output Format" in core_content
        assert "# Custom Instructions Priority" in core_content

        spawn_start = result.index("<spawning-guideline>")
        spawn_end = result.index("</spawning-guideline>")
        spawn_content = result[spawn_start:spawn_end]
        assert "# Execution Strategy (Spawning)" in spawn_content
        assert "# Spawning Rules" in spawn_content


# ---------------------------------------------------------------------------
# build_subagent_prompt()
# ---------------------------------------------------------------------------


class TestBuildSubagentPrompt:

    def test_build_subagent_prompt_with_env(self) -> None:
        result = build_subagent_prompt(
            parent_label="Orchestrator",
            task_desc="Find files",
            depth=1,
            can_spawn=False,
            task_id="task_123",
            label="worker-1",
            env_context={"Date": "Mon Jan 01 2026", "OS": "Darwin"},
        )

        assert "<environment>" in result
        assert "</environment>" in result
        assert "## Environment" not in result
        assert "- **Date**: Mon Jan 01 2026" in result
        assert "- **OS**: Darwin" in result
        assert "Orchestrator" in result
        assert "Find files" in result

    def test_build_subagent_prompt_without_env(self) -> None:
        result = build_subagent_prompt(
            parent_label="Orchestrator",
            task_desc="Search code",
            depth=1,
            can_spawn=False,
            task_id="task_456",
            label="worker-2",
        )

        assert "<environment>" not in result
        assert "Orchestrator" in result
        assert "Search code" in result
        assert "task_456" in result


# ---------------------------------------------------------------------------
# get_subagent_system_prompt()
# ---------------------------------------------------------------------------


class TestGetSubagentSystemPrompt:

    def test_get_subagent_system_prompt(self) -> None:
        result = get_subagent_system_prompt(
            parent_label="RootAgent",
            task_desc="Analyze data",
            depth=2,
            can_spawn=False,
            task_id="task_789",
            label="sub-worker",
        )

        assert "{parent_label}" not in result
        assert "{task_desc}" not in result
        assert "{task_id}" not in result
        assert "{label}" not in result
        assert "RootAgent" in result
        assert "Analyze data" in result
        assert "task_789" in result
        assert "sub-worker" in result


# ---------------------------------------------------------------------------
# DEFAULT_SYSTEM_PROMPT
# ---------------------------------------------------------------------------


class TestDefaultSystemPrompt:

    def test_default_system_prompt_import(self) -> None:
        assert isinstance(DEFAULT_SYSTEM_PROMPT, str)
        assert len(DEFAULT_SYSTEM_PROMPT) > 0
        assert "<core-rules>" in DEFAULT_SYSTEM_PROMPT
        assert "<spawning-guideline>" in DEFAULT_SYSTEM_PROMPT
        assert "# Execution Strategy (Spawning)" in DEFAULT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# _ENV_BLOCK_PATTERN
# ---------------------------------------------------------------------------


class TestEnvBlockPattern:

    def test_env_block_pattern_matches_current_format(self) -> None:
        block = "<environment>\n- **Date**: Mon Jan 01 2026\n- **OS**: Darwin\n</environment>"
        match = _ENV_BLOCK_PATTERN.search(block)
        assert match is not None
        assert match.group(0) == block

    def test_env_block_pattern_rejects_non_env(self) -> None:
        assert _ENV_BLOCK_PATTERN.search("## Available Tools\n- **read**: Read files\n") is None
        assert _ENV_BLOCK_PATTERN.search("## Environment\n- **Date**: test\n") is None
