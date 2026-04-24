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

__all__ = ["delegate", "Tool", "FunctionTool", "AgentResult", "AgentTaskRegistry", "init_logger", "get_logger", "stop_logger"]

DEFAULT_SYSTEM_PROMPT = (
    "你是主Agent，请尽可能构建子代理来并行处理任务。"
    "这不仅可以提升处理效率，还能减少无关信息对上下文的污染，帮助你做出更清晰的决策。"
)

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
    session.add_message("system", system_prompt or DEFAULT_SYSTEM_PROMPT)

    try:
        if config is None:
            config = get_config()

        init_logger(log_dir=config.log_dir)

        # Resolve provider profiles (backward compatible)
        profiles = []
        if config.provider_profiles:
            for p in config.provider_profiles:
                profiles.append(ProviderProfile(**p))
        else:
            # Backward compat: create single profile from legacy .env vars
            profiles.append(ProviderProfile(
                name="default",
                api_key=config.api_key,
                base_url=config.base_url,
                model=config.model,
                rpm_limit=config.rate_limit_rpm,
                tpm_limit=0,
            ))

        # Create per-profile infrastructure
        providers = {}       # profile_name -> LLMProvider
        rate_limiters = {}   # profile_name -> SlidingWindowRateLimiter
        pacers = {}          # profile_name -> AdaptivePacer

        for profile in profiles:
            # Create a Config object for each profile's LLMProvider
            from engine.config import Config as ConfigClass
            provider_config = ConfigClass(
                api_key=profile.api_key,
                base_url=profile.base_url,
                model=profile.model,
                llm_retry_max_attempts=config.llm_retry_max_attempts,
                llm_retry_base_delay=config.llm_retry_base_delay,
            )

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
                )
            pacers[profile.name] = pacer

            # LLM Provider
            providers[profile.name] = LLMProvider(
                provider_config,
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
