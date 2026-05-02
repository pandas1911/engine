"""Runner module — extracted from engine/__init__.py. See engine/__init__.py for re-exports."""

import importlib
import inspect
import logging
import re
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, List, Optional

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
from engine.prompts import build_root_system_prompt

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
                    except Exception as exc:
                        _logger.warning(
                            "Failed to instantiate tool class %s from %s: %s",
                            name, module_name, exc,
                        )
        except Exception as exc:
            _logger.warning(
                "Failed to import custom tool module %s: %s",
                module_name, exc,
            )

    _custom_tools_cache = tools
    return tools


def _refresh_custom_tools() -> None:
    """Clear the custom tools cache, forcing re-discovery on next call."""
    global _custom_tools_cache
    _custom_tools_cache = None


_logger = logging.getLogger(__name__)

_ENV_BLOCK_PATTERN = re.compile(
    r"<env>\s*Today's date:.*?Time zone:.*?\n</env>",
    re.DOTALL,
)


def _refresh_env_block(session: Session, time_provider: TimeProvider) -> None:
    system_msg = None
    for m in session.messages:
        if m.role == "system":
            system_msg = m
            break

    if system_msg is None:
        return

    fresh_block = time_provider.format_system_env_block()
    if _ENV_BLOCK_PATTERN.search(system_msg.content):
        system_msg.content = _ENV_BLOCK_PATTERN.sub(fresh_block, system_msg.content)
    else:
        system_msg.content = f"{system_msg.content}\n\n{fresh_block}"


async def delegate(
    task_description: str,
    system_prompt: Optional[str] = None,
    tools: Optional[List] = None,
    config: Optional[Config] = None,
    session: Optional[Session] = None,
) -> AgentResult:
    """Delegate a task to the agent system."""
    _provided_session = session is not None
    if session is None:
        session = Session(id=f"root_{uuid.uuid4().hex[:8]}", depth=0)

    try:
        if config is None:
            config = get_config()

        time_provider = TimeProvider(timezone_override=config.user_timezone)

        if _provided_session:
            if system_prompt is not None:
                _logger.warning("Both 'session' and 'system_prompt' provided; 'system_prompt' is ignored when 'session' is provided.")
            has_system = any(m.role == "system" for m in session.messages)
            if has_system:
                _refresh_env_block(session, time_provider)
            else:
                base_system_prompt = build_root_system_prompt(
                    include_spawn=config.is_tool_enabled("spawn")
                )
                env_block = time_provider.format_system_env_block()
                full_system_prompt = f"{base_system_prompt}\n\n{env_block}"
                session.add_message("system", full_system_prompt)
        else:
            if system_prompt:
                base_system_prompt = system_prompt
            else:
                base_system_prompt = build_root_system_prompt(
                    include_spawn=config.is_tool_enabled("spawn")
                )
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


async def delegate_stream(
    task_description: str,
    system_prompt: Optional[str] = None,
    tools: Optional[List] = None,
    config: Optional[Config] = None,
    session: Optional[Session] = None,
) -> AsyncGenerator[Any, None]:
    """Streaming counterpart to delegate(). Yields StreamEvent objects."""
    from engine.providers.streaming_models import (
        AgentStartEvent, DoneEvent, ErrorEvent,
    )

    _provided_session = session is not None
    if session is None:
        session = Session(id=f"root_{uuid.uuid4().hex[:8]}", depth=0)

    try:
        if config is None:
            config = get_config()

        time_provider = TimeProvider(timezone_override=config.user_timezone)

        if _provided_session:
            if system_prompt is not None:
                _logger.warning("Both 'session' and 'system_prompt' provided; 'system_prompt' is ignored when 'session' is provided.")
            has_system = any(m.role == "system" for m in session.messages)
            if has_system:
                _refresh_env_block(session, time_provider)
            else:
                base_system_prompt = build_root_system_prompt(
                    include_spawn=config.is_tool_enabled("spawn")
                )
                env_block = time_provider.format_system_env_block()
                full_system_prompt = f"{base_system_prompt}\n\n{env_block}"
                session.add_message("system", full_system_prompt)
        else:
            if system_prompt:
                base_system_prompt = system_prompt
            else:
                base_system_prompt = build_root_system_prompt(
                    include_spawn=config.is_tool_enabled("spawn")
                )
            env_block = time_provider.format_system_env_block()
            full_system_prompt = f"{base_system_prompt}\n\n{env_block}"
            session.add_message("system", full_system_prompt)

        init_logger(log_dir=config.log_dir)

        providers = {}
        rate_limiters = {}
        pacers = {}

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

        enabled_tools = [
            t for t in all_tool_instances
            if config.is_tool_enabled(t.name)
        ]

        if config.is_tool_enabled("spawn"):
            enabled_tools.append(SpawnTool())

        tool_pack = ToolPack(enabled_tools)

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

        yield AgentStartEvent()

        async for event in agent.run_streaming(task_description):
            yield event

        success = agent.state == AgentState.COMPLETED
        yield DoneEvent(data={
            "success": success,
            "content": getattr(agent, '_final_result', '') or "",
        })

    except Exception as e:
        yield ErrorEvent(data={"message": str(e)})
    finally:
        await stop_logger()
