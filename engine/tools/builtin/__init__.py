"""Built-in tools for the agent system."""

from engine.tools.builtin.spawn import SpawnTool

BUILTIN_TOOLS = [SpawnTool]

__all__ = ["SpawnTool", "BUILTIN_TOOLS"]
