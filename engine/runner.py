"""Runner module — extracted from engine/__init__.py. See engine/__init__.py for re-exports."""

import importlib
import inspect
import uuid
from pathlib import Path
from typing import List, Optional

from engine.runtime.agent import Agent
from engine.config import Config, get_config
from engine.providers.llm_provider import LLMProvider
from engine.logging import init_logger, stop_logger
from engine.runtime.agent_models import AgentError, AgentResult, AgentState, ErrorCategory, Session
from engine.runtime.task_registry import AgentTaskRegistry
from engine.tools.base import Tool
from engine.tools.pack import ToolPack
from engine.subagent.spawn import SpawnTool
from engine.safety import LaneConcurrencyQueue, SlidingWindowRateLimiter, AdaptivePacer, APIKeyPool, RetryEngine
from engine.providers.fallback_provider import FallbackLLMProvider
from engine.providers.provider_models import ProviderParams, Lane
from engine.time import TimeProvider

_BASE_PROMPT = """\
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

_SPAWN_PROMPT = """\
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

# Backward-compatible alias: the full prompt with spawn enabled
DEFAULT_SYSTEM_PROMPT = _BASE_PROMPT + "\n" + _SPAWN_PROMPT

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
        if system_prompt:
            base_system_prompt = system_prompt
        else:
            base_system_prompt = _BASE_PROMPT
            if config.is_tool_enabled("spawn"):
                base_system_prompt += "\n" + _SPAWN_PROMPT
        env_block = time_provider.format_system_env_block()
        full_system_prompt = f"{base_system_prompt}\n\n{env_block}"
        session.add_message("system", full_system_prompt)

        init_logger(log_dir=config.log_dir)

        # Build LLMProvider instances — one per provider/model combination
        providers = {}       # composite_key "provider/model" → LLMProvider
        rate_limiters = {}   # provider_name → SlidingWindowRateLimiter
        pacers = {}          # provider_name → AdaptivePacer

        for prov_name, prov_config in config.providers.items():
            limiter = None
            if prov_config.rpm_limit > 0 or prov_config.tpm_limit > 0:
                limiter = SlidingWindowRateLimiter(
                    rpm_limit=prov_config.rpm_limit,
                    tpm_limit=prov_config.tpm_limit,
                    profile_name=prov_name,
                )
            rate_limiters[prov_name] = limiter

            pacer = None
            if config.pacing_enabled:
                pacer = AdaptivePacer(
                    min_interval_ms=config.pacing_min_interval_ms,
                    enabled=True,
                    rpm_limit=prov_config.rpm_limit,
                )
            pacers[prov_name] = pacer

            for model_name, model_params in prov_config.models.items():
                composite_key = f"{prov_name}/{model_name}"
                providers[composite_key] = LLMProvider(
                    provider_params=ProviderParams(
                        api_key=prov_config.api_key,
                        base_url=prov_config.base_url,
                        model=model_name,
                    ),
                    runtime_config=config,
                    model_params=model_params if model_params else None,
                )

        # Build ordered provider list from primary + fallback
        ordered_keys = [config.primary] + config.fallback

        key_pool = APIKeyPool(
            ordered_keys,
            cooldown_initial_ms=config.cooldown_initial_ms,
            cooldown_max_ms=config.cooldown_max_ms,
        )

        shared_retry_engine = RetryEngine(
            max_attempts=config.llm_retry_max_attempts,
            base_delay=config.llm_retry_base_delay,
        )

        ordered_providers = {k: providers[k] for k in ordered_keys}

        llm_provider = FallbackLLMProvider(
            providers=ordered_providers,
            key_pool=key_pool,
            rate_limiters=rate_limiters,
            pacers=pacers,
            retry_engine=shared_retry_engine,
        )

        task_registry = AgentTaskRegistry()

        custom_tools = _discover_custom_tools()
        all_tool_instances = custom_tools + (tools or [])

        # Filter by config.tools (enable/disable)
        enabled_tools = [
            t for t in all_tool_instances
            if config.is_tool_enabled(t.name)
        ]

        # Conditionally add SpawnTool
        if config.is_tool_enabled("spawn"):
            enabled_tools.append(SpawnTool())

        tool_pack = ToolPack(enabled_tools)

        # Create Lane Concurrency Queue
        lane_queue = LaneConcurrencyQueue()
        lane_queue.configure_lane(Lane.MAIN, max_concurrent=config.main_lane_concurrency)
        lane_queue.configure_lane(Lane.SUBAGENT, max_concurrent=config.subagent_lane_concurrency)

        agent = Agent(
            session=session,
            config=config,
            llm_provider=llm_provider,
            task_registry=task_registry,
            tool_pack=tool_pack,
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
