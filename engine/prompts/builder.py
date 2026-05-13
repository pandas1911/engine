"""Prompt layer assembly for the engine framework.

XML-tagged system prompt assembly:
  <core-rules>: base.md — behavioral constraints
  <spawning-guideline>: spawn.md — sub-agent spawning strategy
  <environment>: env context — Date, Timezone, Working Directory, Model, OS
  <available-tools>: tool short_descriptions — enabled tools with descriptions
  <user-instructions>: FRIDAY.md — user-defined behavioral instructions
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PROMPTS_DIR = Path(__file__).parent


def _read_md(filename: str) -> str:
    """Read a markdown file from the prompts directory."""
    return (_PROMPTS_DIR / "templates" / filename).read_text(encoding="utf-8").strip()


# Module-level constants for backward-compatible re-exports
BASE_PROMPT: str = _read_md("base.md")
SPAWN_PROMPT: str = _read_md("spawn.md")


def build_system_prompt(
    include_spawn: bool = False,
    env_context: Optional[Dict[str, str]] = None,
    tool_descriptions: Optional[List[Tuple[str, str]]] = None,
    user_instructions: Optional[str] = None,
) -> str:
    sections = [f"<core-rules>\n{_read_md('base.md')}\n</core-rules>"]

    if include_spawn:
        sections.append(f"<spawning-guideline>\n{_read_md('spawn.md')}\n</spawning-guideline>")

    if env_context:
        lines = ["<environment>"]
        for key, value in env_context.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("</environment>")
        sections.append("\n".join(lines))

    if tool_descriptions:
        lines = ["<available-tools>"]
        for name, desc in tool_descriptions:
            lines.append(f"- **{name}**: {desc}")
        lines.append("</available-tools>")
        sections.append("\n".join(lines))

    if user_instructions:
        sections.append(f"<user-instructions>\n{user_instructions.strip()}\n</user-instructions>")

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
      2. <environment> (if env_context provided)
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
        lines = ["<environment>"]
        for key, value in env_context.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("</environment>")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)
