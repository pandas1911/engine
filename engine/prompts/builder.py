"""Prompt layer assembly for the engine framework.

4-layer system prompt assembly:
  Layer 1: base.md (+ optional spawn.md) — Identity/Rules
  Layer 2: ## Environment (Markdown key-value) — Context
  Layer 3: ## Available Tools (tool short_descriptions) — Tool Awareness
  Layer 4: ## Custom Instructions (FRIDAY.md content) — User Instructions
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PROMPTS_DIR = Path(__file__).parent


def _read_md(filename: str) -> str:
    """Read a markdown file from the prompts directory."""
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


# Module-level constants for backward-compatible re-exports
BASE_PROMPT: str = _read_md("base.md")
SPAWN_PROMPT: str = _read_md("spawn.md")


def build_system_prompt(
    include_spawn: bool = False,
    env_context: Optional[Dict[str, str]] = None,
    tool_descriptions: Optional[List[Tuple[str, str]]] = None,
    user_instructions: Optional[str] = None,
) -> str:
    """Assemble the full system prompt from all 4 layers.

    Layer order:
      1. base.md (+ optional spawn.md)
      2. ## Environment (Markdown key-value)
      3. ## Available Tools (tool short_descriptions)
      4. ## Custom Instructions (FRIDAY.md content)
    """
    sections = [_read_md("base.md")]

    if include_spawn:
        sections.append(_read_md("spawn.md"))

    if env_context:
        lines = ["## Environment"]
        for key, value in env_context.items():
            lines.append(f"- **{key}**: {value}")
        sections.append("\n".join(lines))

    if tool_descriptions:
        lines = ["## Available Tools"]
        for name, desc in tool_descriptions:
            lines.append(f"- **{name}**: {desc}")
        sections.append("\n".join(lines))

    if user_instructions:
        sections.append(f"## Custom Instructions\n\n{user_instructions.strip()}")

    return "\n\n".join(sections)


# Backward-compatible aliases
def build_root_system_prompt(include_spawn: bool) -> str:
    """Backward-compatible alias for build_system_prompt().

    Returns only Layer 1 (base + optional spawn) without runtime layers.
    """
    return build_system_prompt(include_spawn=include_spawn)


DEFAULT_SYSTEM_PROMPT: str = build_root_system_prompt(include_spawn=True)


def get_subagent_system_prompt(
    parent_label: str,
    task_desc: str,
    depth: int,
    can_spawn: bool,
    task_id: str,
    label: str = "",
) -> str:
    """Build the sub-agent system prompt from subagent.md template.

    Uses .format() for variable substitution (matching current convention).
    """
    template = _read_md("subagent.md")
    return template.format(
        parent_label=parent_label,
        task_desc=task_desc,
        task_id=task_id,
        label=label,
    )


def build_subagent_prompt(
    parent_label: str,
    task_desc: str,
    depth: int,
    can_spawn: bool,
    task_id: str,
    label: str = "",
    env_context: Optional[Dict[str, str]] = None,
) -> str:
    """Assemble the full sub-agent system prompt with env layer.

    Called by SubAgentManager.spawn() — single entry point
    for sub-agent prompt assembly.

    Layers:
      1. subagent.md template (.format substituted)
      2. ## Environment (if env_context provided)
    """
    sections = [get_subagent_system_prompt(
        parent_label=parent_label,
        task_desc=task_desc,
        depth=depth,
        can_spawn=can_spawn,
        task_id=task_id,
        label=label,
    )]

    if env_context:
        lines = ["## Environment"]
        for key, value in env_context.items():
            lines.append(f"- **{key}**: {value}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)
