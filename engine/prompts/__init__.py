"""Backward-compatible re-exports for the engine.prompts package.

All symbols from the old flat prompts.py are re-exported here
so that existing `from engine.prompts import X` statements continue to work.

Dead code removed: get_child_results_prompt, get_child_results_empty_warning
"""

from engine.prompts.builder import (
    BASE_PROMPT,
    SPAWN_PROMPT,
    build_root_system_prompt,
    DEFAULT_SYSTEM_PROMPT,
    get_subagent_system_prompt,
    build_system_prompt,
    build_subagent_prompt,
)

from engine.prompts.runtime import (
    get_summary_warning,
    get_emergency_summary_prompt,
    get_spawn_confirmation,
    get_concurrency_timeout_rejection,
)

__all__ = [
    "BASE_PROMPT",
    "SPAWN_PROMPT",
    "build_root_system_prompt",
    "build_system_prompt",
    "build_subagent_prompt",
    "DEFAULT_SYSTEM_PROMPT",
    "get_subagent_system_prompt",
    "get_summary_warning",
    "get_emergency_summary_prompt",
    "get_spawn_confirmation",
    "get_concurrency_timeout_rejection",
]
