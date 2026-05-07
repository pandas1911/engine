from typing import Any, Dict, List, Optional

from engine.tools.base import Tool, ToolRegistry, ToolRegistrationError


# Tools that only the root agent (depth=0) should have access to.
_ROOT_ONLY_TOOLS = {"spawn", "list_children", "read_session"}


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

        Root-only tools (spawn, list_children, read_session) are hidden from
        sub-agents (depth >= 1) since depth=1 is enforced at architecture level.
        """
        all_schemas = self._registry.get_schemas()

        if session is not None and session.depth >= 1:
            return [s for s in all_schemas if s["function"]["name"] not in _ROOT_ONLY_TOOLS]

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
