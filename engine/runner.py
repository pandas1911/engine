"""Runner module — Infrastructure, Engine, and SessionManager."""

import asyncio
import importlib
import inspect
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from engine.runtime.agent import Agent
from engine.config import Config, get_config
from engine.providers.llm_provider import LLMProvider
from engine.logging import init_logger
from engine.runtime.agent_models import AgentError, AgentResult, AgentState, ErrorCategory, Session
from engine.runtime.task_registry import AgentTaskRegistry
from engine.tools.base import Tool
from engine.tools.pack import ToolPack
from engine.tools.builtin import BUILTIN_TOOLS
from engine.safety import SlidingWindowRateLimiter, APIKeyPool, RetryEngine
from engine.providers.fallback_provider import FallbackLLMProvider
from engine.providers.provider_models import ProviderParams
from engine.time import TimeProvider
from engine.prompts.builder import build_system_prompt
from engine.prompts.env_builder import build_env_context
from engine.streaming_handler import SSEStreamingHandler

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
    r"<environment>\n(?:- \*\*.*?\*\*: .*?\n)*</environment>",
)

_active_sessions: Dict[str, "SessionManager"] = {}


class Infrastructure:
    """One-time infrastructure. Holds providers, rate limiters, tools, etc.
    Created once at server startup (or lazily on first delegate() call).
    All SessionManagers share the same Infrastructure instance.
    """

    def __init__(self, config: Optional[Config] = None):
        if config is None:
            config = get_config()
        self.config = config
        self.time_provider = TimeProvider(timezone_override=config.user_timezone)

        init_logger(config=config)

        # --- Build providers ---
        self.providers: Dict[str, LLMProvider] = {}
        self.rate_limiters: Dict[str, SlidingWindowRateLimiter] = {}

        for prov_name, prov_config in config.providers.items():
            limiter = None
            if prov_config.rpm_limit > 0 or prov_config.tpm_limit > 0:
                limiter = SlidingWindowRateLimiter(
                    rpm_limit=prov_config.rpm_limit,
                    tpm_limit=prov_config.tpm_limit,
                    profile_name=prov_name,
                    pacing_enabled=config.pacing_enabled,
                    min_interval_ms=config.pacing_min_interval_ms,
                )
            self.rate_limiters[prov_name] = limiter

            for model_name, model_params in prov_config.models.items():
                composite_key = f"{prov_name}/{model_name}"
                self.providers[composite_key] = LLMProvider(
                    provider_params=ProviderParams(
                        api_key=prov_config.api_key,
                        base_url=prov_config.base_url,
                        model=model_name,
                    ),
                    runtime_config=config,
                    model_params=model_params if model_params else None,
                )

        ordered_keys = [config.primary] + config.fallback
        self.key_pool = APIKeyPool(
            ordered_keys,
            cooldown_initial_ms=config.cooldown_initial_ms,
            cooldown_max_ms=config.cooldown_max_ms,
        )
        self.retry_engine = RetryEngine(
            max_attempts=config.llm_retry_max_attempts,
            base_delay=config.llm_retry_base_delay,
        )
        # Build concurrency guards per provider
        concurrency_guards: Dict[str, asyncio.Semaphore] = {}
        for prov_name, prov_config in config.providers.items():
            if prov_config.max_concurrent_requests > 0:
                concurrency_guards[prov_name] = asyncio.Semaphore(
                    prov_config.max_concurrent_requests
                )

        ordered_providers = {k: self.providers[k] for k in ordered_keys}
        self.llm_provider = FallbackLLMProvider(
            providers=ordered_providers,
            key_pool=self.key_pool,
            rate_limiters=self.rate_limiters,
            retry_engine=self.retry_engine,
            concurrency_guards=concurrency_guards,
        )

        # --- Build tools ---
        self.tool_pack = self._build_tool_pack(config)

        # --- Build session store (workspace-based) ---
        from engine.session_store import SessionStore
        session_root = str(config.get_workspace_path() / "sessions")
        self.session_store = SessionStore(root_dir=session_root)

    def _build_tool_pack(self, config: Config) -> ToolPack:
        custom_tools = _discover_custom_tools()
        builtin_tool_instances = [cls() for cls in BUILTIN_TOOLS]
        all_tool_instances = builtin_tool_instances + custom_tools
        enabled_tools = [
            t for t in all_tool_instances
            if config.is_tool_enabled(t.name)
        ]
        return ToolPack(enabled_tools)


class Engine:
    """Unified entry point for the agent system.
    Owns Infrastructure, provides delegate() and create_session().
    Singleton via Engine.get().
    """

    _instance: Optional["Engine"] = None

    def __init__(self, config: Optional[Config] = None):
        self._infra = Infrastructure(config)

    @classmethod
    def get(cls, config: Optional[Config] = None) -> "Engine":
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """For testing — clear the singleton."""
        cls._instance = None

    async def delegate(
        self,
        task_description: str,
        tools: Optional[List] = None,
        config: Optional[Config] = None,
        session: Optional[Session] = None,
        event_callback: Optional[Callable[[str, Any], None]] = None,
    ) -> AgentResult:
        """Delegate a task to the agent system."""
        mgr = None
        try:
            mgr = self.create_session(
                session=session,
                event_callback=event_callback,
            )
            return await mgr.start(task_description)
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
            if mgr:
                mgr.unregister()

    def create_session(
        self,
        session: Optional[Session] = None,
        event_callback: Optional[Callable[[str, Any], None]] = None,
    ) -> "SessionManager":
        """Create a SessionManager for a conversation."""
        return SessionManager(
            infra=self._infra,
            session=session,
            event_callback=event_callback,
        )


class SessionManager:
    """Per-conversation manager. Creates and owns the root Agent.
    Core logic is ~10 lines:
    - start(): run initial message, wait for completion
    """

    def __init__(
        self,
        infra: "Infrastructure",
        session: Optional[Session] = None,
        event_callback: Optional[Callable[[str, Any], None]] = None,
    ):
        self.infra = infra
        self.event_callback = event_callback

        # --- Session setup ---
        if session is None:
            session = Session(id=f"root_{uuid.uuid4().hex[:8]}", depth=0)
        self.session = session

        self._ensure_system_prompt()

        # --- Unified event queue (passed to Agent, shared with Branch B) ---
        self._event_queue: list = []

        # --- Agent creation ---
        self.streaming_handler = (
            SSEStreamingHandler(event_callback) if event_callback else None
        )
        self.task_registry = AgentTaskRegistry()

        self.agent = Agent(
            session=self.session,
            config=infra.config,
            llm_provider=infra.llm_provider,
            task_registry=self.task_registry,
            tool_pack=infra.tool_pack,
            streaming_handler=self.streaming_handler,
            event_queue=self._event_queue,
        )

        # --- Session persistence ---
        self.session_store = infra.session_store
        self.session_store.create_root(self.session.id)
        self.agent.session_store = self.session_store

        # Only create a new file for brand-new sessions.
        # Restored sessions already have their file on disk — calling create_file
        # would overwrite it with a header-only file, wiping conversation history.
        existing = self.session_store._find_file("main")
        if existing is None:
            self.session_store.create_file("main", self.session)

        self.session._on_message_added = (
            lambda msg: self.session_store.append_line("main", msg)
        )

    def _ensure_system_prompt(self) -> None:
        has_system = any(m.role == "system" for m in self.session.messages)
        if has_system:
            self._refresh_env_block()
        else:
            # Resolve workspace directory
            workspace_dir = str(self.infra.config.get_workspace_path())

            # Build env context
            env_context = build_env_context(
                time_provider=self.infra.time_provider,
                workspace_dir=workspace_dir,
                model_name=self.infra.config.primary,
            )

            # Collect tool short_descriptions
            tool_descs = [
                (t.name, t.short_description)
                for t in self.infra.tool_pack._registry._tools.values()
                if hasattr(t, 'short_description') and t.short_description
            ] if self.infra.tool_pack else []

            # Scan FRIDAY.md from workspace directory
            user_instructions = None
            friday_path = self.infra.config.get_workspace_path() / "FRIDAY.md"
            if friday_path.exists():
                user_instructions = friday_path.read_text(encoding="utf-8")

            # Assemble full system prompt
            full_prompt = build_system_prompt(
                include_spawn=self.infra.config.is_tool_enabled("spawn"),
                env_context=env_context,
                tool_descriptions=tool_descs if tool_descs else None,
                user_instructions=user_instructions,
            )
            self.session.add_message("system", full_prompt)

    def _refresh_env_block(self) -> None:
        system_msg = next(
            (m for m in self.session.messages if m.role == "system"), None
        )
        if system_msg is None:
            return

        workspace_dir = str(self.infra.config.get_workspace_path())
        env_context = build_env_context(
            time_provider=self.infra.time_provider,
            workspace_dir=workspace_dir,
            model_name=self.infra.config.primary,
        )
        fresh_lines = ["<environment>"]
        for key, value in env_context.items():
            fresh_lines.append(f"- **{key}**: {value}")
        fresh_lines.append("</environment>")
        fresh_block = "\n".join(fresh_lines)

        if _ENV_BLOCK_PATTERN.search(system_msg.content):
            system_msg.content = _ENV_BLOCK_PATTERN.sub(fresh_block, system_msg.content)
        else:
            system_msg.content = f"{system_msg.content}\n\n{fresh_block}"

    # --- Core execution ---

    async def start(self, message: str) -> AgentResult:
        """First message: register agent, run, wait for completion."""
        await self.task_registry.register(
            task_id=self.agent.task_id,
            session_id=self.session.id,
            description="root task",
            agent=self.agent,
            depth=0,
        )
        self._register()

        await self.agent.run(message)

        if not self.agent._completion_event.is_set():
            await self.agent._completion_event.wait()

        return self._build_result()

    # --- Lifecycle ---

    def _build_result(self) -> AgentResult:
        success = self.agent.state == AgentState.COMPLETED
        return AgentResult(
            content=self.agent._final_result or "",
            session=self.session,
            success=success,
            error=None if success else self.agent._error_info,
        )

    def _register(self) -> None:
        _active_sessions[self.session.id] = self

    def unregister(self) -> None:
        _active_sessions.pop(self.session.id, None)
