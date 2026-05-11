"""Runtime prompt functions for the engine framework.

Dynamic prompt generation for warnings, confirmations, and rejections.
Pure functions with no engine.* imports.
"""


def get_summary_warning(remaining_iterations: int) -> str:
    """Warning injected when approaching the tool call iteration limit."""
    return (
        "[System Notice] You have {} tool call iteration(s) remaining. "
        "Please stop making tool calls and provide your final comprehensive answer "
        "based on all data you have collected so far. Do NOT make any more tool calls."
    ).format(remaining_iterations)


def get_emergency_summary_prompt() -> str:
    """Emergency summary prompt forcing a final answer without tool calls."""
    return (
        "[System] You have exhausted all available tool call iterations. "
        "You MUST now provide a comprehensive final answer based on all the "
        "data and results you have gathered. Structure your answer clearly."
    )


def get_spawn_confirmation(task_id: str, label: str) -> str:
    """Confirmation message returned after spawning a child agent."""
    return (
        "━━━━ Spawned Task ━━━━\n"
        "Task ID: {task_id}\n"
        "Agent Label: {label}\n"
        "\n"
        "Sub-agent is now executing in the background. Upon completion, you will be automatically re-activated and receive a full result report. You may proceed with other independent tasks or simply end your current turn."
    ).format(task_id=task_id, label=label)


def get_concurrency_timeout_rejection(
    task_desc: str,
    label: str,
    active: int,
    max_concurrent: int,
    timeout: float,
) -> str:
    """Rejection message when spawn fails due to concurrency timeout."""
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
