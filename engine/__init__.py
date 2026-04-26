"""engine - AI Agent automation package."""

import importlib
import inspect
import uuid
from pathlib import Path
from typing import List, Optional

from engine.runtime.agent import Agent
from engine.config import Config, get_config
from engine.providers.llm_provider import LLMProvider
from engine.logging import init_logger, get_logger, stop_logger
from engine.runtime.agent_models import AgentError, AgentResult, AgentState, ErrorCategory, Session
from engine.runtime.task_registry import AgentTaskRegistry
from engine.tools.base import Tool, FunctionTool
from engine.safety import LaneConcurrencyQueue, SlidingWindowRateLimiter, AdaptivePacer, APIKeyPool, RetryEngine
from engine.providers.fallback_provider import FallbackLLMProvider
from engine.providers.provider_models import ProviderProfile, Lane
from engine.time import TimeProvider

__all__ = ["delegate", "Tool", "FunctionTool", "AgentResult", "AgentTaskRegistry", "init_logger", "get_logger", "stop_logger"]

DEFAULT_SYSTEM_PROMPT = """\
# Root Agent

You are the root orchestrator agent. Your job is to decompose tasks, dispatch work, and synthesize results.

## Execution Strategy

1. **Decompose first** — Break the task into independent subtasks. Assign each to a child agent via `spawn`.
2. **Parallel over sequential** — If subtasks have no dependencies, dispatch them all in one turn.
3. **Handle simple tasks yourself** — If a task is trivial (single-step, no research needed), do it directly rather than spawning overhead.
4. **Use tools proactively** — When tools are available, prefer using them over reasoning from incomplete knowledge. Vary your approach if a tool returns weak or empty results.
5. **Ground your response in evidence** — Strictly base your answers and next actions on tool results and child agent reports. Never fabricate information or speculate beyond what the evidence supports.
6. **Iterate after synthesis** — After child agents report back, evaluate whether the results are sufficient to complete the task. If so, synthesize and respond. If not, plan and dispatch further work.

## Spawning Rules

- One `spawn` call = one focused subtask with clear completion criteria.
- Include sufficient context in the task description — the child agent starts isolated.
- Respect the depth limit: at maximum depth, complete the task yourself.
- Do NOT spawn a child for tasks that require a single tool call you can make yourself.

## Output Format

When the task specifies an output format, follow it exactly. The guidelines below apply when no format is specified.

Be concise and structured:
- Start with the direct answer or conclusion.
- Follow with supporting details only when they add value.
- No filler, no meta-commentary ("I have completed...", "Here is...").
- For multi-part tasks, use clear headings or bullet lists.
"""

_custom_tools_cache: Optional[List] = None


def _discover_custom_tools() -> List:
    """Discover and cache custom tools from engine/tools/custom/."""
    global _custom_tools_cache
    if _custom_tools_cache is not None:
        return _custom_tools_cache

    tools = []
    custom_dir = Path(__file__).parent / "tools" / "custom"

    if not custom_dir.exists():
        _custom_tools_cache = tools
        return tools

    for py_file in custom_dir.glob("*.py"):
        if py_file.name.startswith("_") or py_file.name == "__init__.py":
            continue

        module_name = f"engine.tools.custom.{py_file.stem}"
        try:
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, Tool) and obj is not Tool:
                    try:
                        tools.append(obj())
                    except Exception:
                        pass
        except Exception:
            pass

    _custom_tools_cache = tools
    return tools


def _refresh_custom_tools() -> None:
    """Clear the custom tools cache, forcing re-discovery on next call."""
    global _custom_tools_cache
    _custom_tools_cache = None


async def delegate(
    task_description: str,
    system_prompt: Optional[str] = None,
    tools: Optional[List] = None,
    config: Optional[Config] = None,
) -> AgentResult:
    """Delegate a task to the agent system."""
    session = Session(id=f"root_{uuid.uuid4().hex[:8]}", depth=0)

    try:
        if config is None:
            config = get_config()

        # Layer 1: Inject static time info into system prompt
        time_provider = TimeProvider(timezone_override=config.user_timezone)
        base_system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        env_block = time_provider.format_system_env_block()
        full_system_prompt = f"{base_system_prompt}\n\n{env_block}"
        session.add_message("system", full_system_prompt)

        init_logger(log_dir=config.log_dir)

        profiles = [ProviderProfile(**p) for p in config.provider_profiles]

        # Create per-profile infrastructure
        providers = {}       # profile_name -> LLMProvider
        rate_limiters = {}   # profile_name -> SlidingWindowRateLimiter
        pacers = {}          # profile_name -> AdaptivePacer

        for profile in profiles:
            # Rate limiter (RPM + TPM)
            limiter = None
            if profile.rpm_limit > 0 or profile.tpm_limit > 0:
                limiter = SlidingWindowRateLimiter(
                    rpm_limit=profile.rpm_limit,
                    tpm_limit=profile.tpm_limit,
                    profile_name=profile.name,
                )
            rate_limiters[profile.name] = limiter

            # Adaptive pacer
            pacer = None
            if config.pacing_enabled:
                pacer = AdaptivePacer(
                    min_interval_ms=config.pacing_min_interval_ms,
                    enabled=True,
                    rpm_limit=profile.rpm_limit,
                )
            pacers[profile.name] = pacer

            # LLM Provider
            providers[profile.name] = LLMProvider(
                api_key=profile.api_key,
                base_url=profile.base_url,
                model=profile.model,
                config=config,
                retry_engine=RetryEngine(
                    max_attempts=config.llm_retry_max_attempts,
                    base_delay=config.llm_retry_base_delay,
                ),
            )

        # Create shared components
        key_pool = APIKeyPool(
            profiles,
            cooldown_initial_ms=config.cooldown_initial_ms,
            cooldown_max_ms=config.cooldown_max_ms,
        )

        shared_retry_engine = RetryEngine(
            max_attempts=config.llm_retry_max_attempts,
            base_delay=config.llm_retry_base_delay,
        )

        # Create FallbackProvider
        llm_provider = FallbackLLMProvider(
            providers=providers,
            key_pool=key_pool,
            rate_limiters=rate_limiters,
            pacers=pacers,
            retry_engine=shared_retry_engine,
        )

        task_registry = AgentTaskRegistry()

        custom_tools = _discover_custom_tools()
        all_tools = custom_tools + (tools or [])

        # Create Lane Concurrency Queue
        lane_queue = LaneConcurrencyQueue()
        lane_queue.configure_lane(Lane.MAIN, max_concurrent=config.main_lane_concurrency)
        lane_queue.configure_lane(Lane.SUBAGENT, max_concurrent=config.subagent_lane_concurrency)

        agent = Agent(
            session=session,
            config=config,
            task_registry=task_registry,
            llm_provider=llm_provider,
            tools=all_tools,
            lane_queue=lane_queue,
        )

        await task_registry.register(
            task_id=agent.task_id,
            session_id=session.id,
            description="root task",
            parent_agent=None,
            agent=agent,
            depth=0,
        )

        await agent.run(task_description)

        if agent.state not in (AgentState.COMPLETED, AgentState.ERROR):
            await agent._completion_event.wait()

        success = agent.state == AgentState.COMPLETED
        return AgentResult(
            content=agent._final_result or "",
            session=session,
            success=success,
            error=None if success else agent._error_info,
        )
    except Exception as e:
        return AgentResult(
            content="",
            session=session,
            success=False,
            error=AgentError(
                category=ErrorCategory.INTERNAL_ERROR,
                message=str(e),
                exception_type=type(e).__name__,
            ),
        )
    finally:
        await stop_logger()
