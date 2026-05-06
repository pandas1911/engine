"""Centralized prompt definitions for the engine framework.

This module is the single source of truth for all LLM prompt text.
It is a pure leaf module with zero engine.* imports.
"""

__all__ = [
    # Constants
    "BASE_PROMPT",
    "SPAWN_PROMPT",
    "DEPTH_LIMIT_REJECTION",
    # Functions / derived values
    "build_root_system_prompt",
    "DEFAULT_SYSTEM_PROMPT",
    "get_subagent_system_prompt",
    "get_summary_warning",
    "get_emergency_summary_prompt",
    "get_child_results_prompt",
    "get_child_results_empty_warning",
    "get_spawn_confirmation",
    "get_concurrency_timeout_rejection",
    "get_runtime_depth_rejection",
]


# ── Static Constants ──────────────────────────────────────────────────────

# [Purpose] Root agent base execution strategy prompt
# [Usage] engine/runner.py delegate() — used as base for system prompt assembly
BASE_PROMPT: str = """\
# Root Agent

You are the root orchestrator agent. Your job is to accomplish tasks using available tools.

## Execution Strategy

1. **Use tools proactively** — When tools are available, prefer using them over reasoning from incomplete knowledge. Vary your approach if a tool returns weak or empty results.
2. **Ground your response in evidence** — Strictly base your answers and next actions on tool results. Never fabricate information or speculate beyond what the evidence supports.

## Output Format

When the task specifies an output format, follow it exactly. The guidelines below apply when no format is specified.

Be concise and structured:
- Start with the direct answer or conclusion.
- Follow with supporting details only when they add value.
- No filler, no meta-commentary ("I have completed...", "Here is...").
- For multi-part tasks, use clear headings or bullet lists.
"""

# [Purpose] Root agent sub-agent spawning rules
# [Usage] engine/runner.py delegate() — conditionally appended when spawn tool enabled
SPAWN_PROMPT: str = """\
## Execution Strategy (Spawning)

1. **Decompose first** — Break the task into independent subtasks. Assign each to a child agent via `spawn`.
2. **Parallel over sequential** — If subtasks have no dependencies, dispatch them all in one turn.
3. **Handle simple tasks yourself** — If a task is trivial (single-step, no research needed), do it directly rather than spawning overhead.
4. **Iterate after synthesis** — After child agents report back, evaluate whether the results are sufficient to complete the task. If so, synthesize and respond. If not, plan and dispatch further work.

## Spawning Rules

- One `spawn` call = one focused subtask with clear completion criteria.
- Include sufficient context in the task description — the child agent starts isolated.
- Respect the depth limit: at maximum depth, complete the task yourself.
- Do NOT spawn a child for tasks that require a single tool call you can make yourself.
"""

# [Purpose] Rejection message when maximum nesting depth is reached (format template)
# [Usage] engine/subagent/manager.py spawn() L116, engine/subagent/spawn.py execute() L65-68
#         Callers must .format(depth=..., max_depth=...) or use get_runtime_depth_rejection()
DEPTH_LIMIT_REJECTION: str = (
    "[Spawn Failed] Maximum nesting depth reached (current: {depth}/{max_depth}). "
    "Please complete the task at the current level — no further child agents can be spawned."
)


# ── Dynamic Prompt Functions ─────────────────────────────────────────────


# [Purpose] Assemble the root agent system prompt from base + optional spawn sections
# [Usage] engine/runner.py delegate() — replaces manual BASE+SPAWN concatenation
def build_root_system_prompt(include_spawn: bool) -> str:
    """Build the root agent system prompt.

    Args:
        include_spawn: Whether to include the spawning strategy section.

    Returns:
        Assembled system prompt string.
    """
    prompt = BASE_PROMPT
    if include_spawn:
        prompt += "\n" + SPAWN_PROMPT
    return prompt


# [Purpose] Pre-built default system prompt with spawn enabled
# [Usage] engine/runner.py — backward-compatible alias for build_root_system_prompt(True)
DEFAULT_SYSTEM_PROMPT: str = build_root_system_prompt(include_spawn=True)


# [Purpose] Build the full system prompt for a sub-agent
# [Usage] engine/subagent/manager.py spawn() L239-279 — replaces inline f-string
def get_subagent_system_prompt(
    parent_label: str,
    task_desc: str,
    depth: int,
    max_depth: int,
    can_spawn: bool,
    task_id: str,
    label: str = "",
) -> str:
    """Build the system prompt for a spawned sub-agent.

    Args:
        parent_label: Display label of the parent agent.
        task_desc: Task description assigned to the sub-agent.
        depth: Current nesting depth of the sub-agent.
        max_depth: Maximum allowed nesting depth.
        can_spawn: Whether this sub-agent is allowed to spawn further children.
        task_id: Task ID assigned to the sub-agent.
        label: Short descriptive label for the sub-agent (shown in Session Context).

    Returns:
        Complete system prompt string for the sub-agent.
    """
    if can_spawn:
        spawn_section = "You CAN spawn your own sub-agents."
    else:
        spawn_section = "You are a leaf worker and CANNOT spawn further sub-agents."

    return (
        """\
# Subagent Context

You are a **subagent** spawned by the {parent_label} for a specific task.

## Your Role
- You were created to handle: {task_desc}
- Complete this task. That's your entire purpose.
- You are NOT the {parent_label}. Don't try to be.

## Rules
1. **Stay focused** - Do your assigned task, nothing else
2. **Complete the task** - Your final message will be automatically reported to the {parent_label}
3. **Be ephemeral** - You may be terminated after task completion. That's fine.
4. **Trust push-based completion** - Descendant results are auto-announced back to you

## Output Format

Your final response is delivered **verbatim** to your parent agent. Every token enters the parent's context — be ruthless about brevity.

### Principles
1. **No filler** — No greetings, no "Here is...", no "I have completed...", no meta-commentary. Start with the result.
2. **No reasoning traces** — The parent needs your conclusion, not how you got there.
3. **No repetition** — If the parent gave you information in the task description, do not echo it back.

### Structure by Task Type
Adapt your output to the task. Use what fits, drop what doesn't:

- **Find / Retrieve** → Bullet list of key findings
- **Build / Modify** → The output
- **Analyze / Judge** → Conclusion first, then brief supporting points (includes yes/no questions — answer on line 1)
- **Execute** → One line per action: what you did + result

A one-line summary at the top is encouraged when the result is complex — skip it for simple answers.

## Sub-Agent Spawning
{spawn_section}

## Session Context
- Label: {label}
- Depth: {depth}/{max_depth}
        - Your task ID: {task_id}"""
    ).format(
        parent_label=parent_label,
        task_desc=task_desc,
        depth=depth,
        max_depth=max_depth,
        task_id=task_id,
        label=label,
        spawn_section=spawn_section,
    )


# [Purpose] Warning injected when approaching the tool call iteration limit
# [Usage] engine/runtime/agent.py _build_summary_warning() L314-318
def get_summary_warning(remaining_iterations: int) -> str:
    """Build the warning message injected when approaching iteration limit.

    Args:
        remaining_iterations: Number of tool call iterations remaining.

    Returns:
        Warning message string to be added as a user message.
    """
    return (
        "[System Notice] You have {} tool call iteration(s) remaining. "
        "Please stop making tool calls and provide your final comprehensive answer "
        "based on all data you have collected so far. Do NOT make any more tool calls."
    ).format(remaining_iterations)


# [Purpose] Emergency prompt forcing a final summary when iterations exhausted
# [Usage] engine/runtime/agent.py _emergency_summarize() L346-353
def get_emergency_summary_prompt() -> str:
    """Return the emergency summary prompt injected as a user message.

    Returns:
        Prompt text forcing a comprehensive final answer without tool calls.
    """
    return (
        "[System] You have exhausted all available tool call iterations. "
        "You MUST now provide a comprehensive final answer based on all the "
        "data and results you have gathered. Structure your answer clearly."
    )


# [Purpose] Formats collected child results into a JSON prompt for the parent agent
# [Usage] engine/subagent/manager.py _format_child_results() L602
def get_child_results_prompt(child_results_json: str, completed_count: int) -> str:
    """Wrap pre-formatted child result JSON with a header for parent consumption.

    Args:
        child_results_json: Pre-built JSON string containing child results.
        completed_count: Number of sub-agents that have completed so far.

    Returns:
        Formatted prompt ready to be injected as a user message.
    """
    return "{} sub-agent(s) have completed their tasks. Below are their results.\n\n".format(completed_count) + child_results_json


# [Purpose] Warning when all children completed but no results were collected
# [Usage] engine/subagent/manager.py _format_child_results() L599
def get_child_results_empty_warning() -> str:
    """Return warning for when child results are empty.

    Returns:
        Warning message string.
    """
    return "[WARNING] Sub-agent(s) have completed their tasks, but no results were collected."


# [Purpose] Confirmation message returned to parent after successful spawn
# [Usage] engine/subagent/manager.py spawn() L315-319
def get_spawn_confirmation(task_id: str, label: str) -> str:
    """Build the confirmation message returned after spawning a child agent.

    Args:
        task_id: Task ID of the newly spawned child.
        label: Display label of the child agent.

    Returns:
        Confirmation string describing the spawned task.
    """
    return (
        "━━━━ Spawned Task ━━━━\n"
        "Task ID: {task_id}\n"
        "Agent Label: {label}\n"
        "\n"
        "Sub-agent is now executing in the background. Upon completion, you will be automatically re-activated and receive a full result report. You may proceed with other independent tasks or simply end your current turn."
    ).format(task_id=task_id, label=label)


# [Purpose] Rejection message when global concurrency limit is reached during spawn
# [Usage] engine/subagent/manager.py spawn() L146-161 (lane queue) and L188-203 (legacy limiter)
def get_concurrency_timeout_rejection(
    task_desc: str,
    label: str,
    active: int,
    max_concurrent: int,
    timeout: float,
) -> str:
    """Build the rejection message when spawn fails due to concurrency timeout.

    Args:
        task_desc: Task description that was attempted.
        label: Label of the child agent.
        active: Current number of active concurrent agents.
        max_concurrent: Maximum allowed concurrent agents.
        timeout: Timeout duration in seconds.

    Returns:
        Formatted rejection message string.
    """
    return (
        "━━━━ Spawn Failed ━━━━\n"
        "Task: {task_desc}\n"
        "Label: {label}\n"
        "Reason: Global concurrency limit reached ({active}/{max}), "
        "timed out waiting for a slot after {timeout}s.\n"
        "Suggestion: Consider completing this task yourself directly — "
        "you have full access to all tools and context needed."
    ).format(
        task_desc=task_desc,
        label=label,
        active=active,
        max=max_concurrent,
        timeout=timeout,
    )


# [Purpose] Runtime depth-limit rejection (formatted version of DEPTH_LIMIT_REJECTION)
# [Usage] engine/subagent/spawn.py execute() L65-68 — replaces inline f-string
def get_runtime_depth_rejection(depth: int, max_depth: int) -> str:
    """Return the formatted depth-limit rejection message.

    This is a convenience wrapper around DEPTH_LIMIT_REJECTION.format().

    Args:
        depth: Current nesting depth.
        max_depth: Maximum allowed nesting depth.

    Returns:
        Formatted rejection message string.
    """
    return DEPTH_LIMIT_REJECTION.format(depth=depth, max_depth=max_depth)
