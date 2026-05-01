from typing import Any, Dict, List, Optional

from engine.tools.base import Tool, ToolRegistry, ToolRegistrationError
from engine.config import get_config


class ToolPack:
    """Immutable view over ToolRegistry with context-aware schema filtering."""

    def __init__(self, tools: List[Tool]):
        self._registry = ToolRegistry()
        for tool in tools:
            if not tool.name or not tool.name.strip():
                raise ToolRegistrationError("Tool name cannot be empty")
            if tool.name in self._registry:
                raise ToolRegistrationError(
                    f"Tool '{tool.name}' is already registered"
                )
            self._registry.register(tool)

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name. Returns None if not found."""
        return self._registry.get(name)

    def get_schemas(self, session=None) -> List[Dict[str, Any]]:
        """Get OpenAI function calling schemas, filtered by session depth.

        Config is obtained via get_config() internally.
        If session is provided and its depth >= config.max_depth, 'spawn' schema is filtered out.
        """
        all_schemas = self._registry.get_schemas()

        if session is not None:
            config = get_config()
            if session.depth >= config.max_depth:
                return [s for s in all_schemas if s["function"]["name"] != "spawn"]

        return all_schemas

    def release_spawn(self, agent_task_id: str) -> None:
        """Forward release to SpawnTool if present."""
        spawn_tool = self._registry.get("spawn")
        if spawn_tool is not None and hasattr(spawn_tool, "release"):
            spawn_tool.release(agent_task_id)

    def __len__(self) -> int:
        return len(self._registry)

    def __contains__(self, name: str) -> bool:
        return name in self._registry
