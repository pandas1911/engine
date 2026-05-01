"""Base tool system for the Agent.

This module contains the Tool ABC, FunctionTool, ToolRegistry, and related exceptions.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional


class Tool(ABC):
    """Base class for all tools.

    Tools are callable functions that agents can use to perform actions.
    """

    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}

    @abstractmethod
    async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Execute the tool with given arguments.

        Args:
            arguments: Tool arguments from LLM
            context: Execution context (session, config, parent_agent, etc.)

        Returns:
            Result string to be passed back to the LLM
        """
        pass


class ToolRegistrationError(Exception):
    """Raised when tool registration fails (duplicate name, empty name, etc.)"""
    pass


class FunctionTool(Tool):
    """Wrap a plain function as a Tool. Supports both sync and async functions."""

    def __init__(self, name: str, description: str, parameters: Dict[str, Any], fn: Callable):
        if not name or not name.strip():
            raise ToolRegistrationError("Tool name cannot be empty")
        self.name = name
        self.description = description
        self.parameters = parameters
        self._fn = fn

    async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
        import asyncio
        result = self._fn(arguments, context)
        if asyncio.iscoroutine(result):
            result = await result
        return str(result)


class ToolRegistry:
    """Central registry for managing tools. Pure storage — no business logic."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Raises ToolRegistrationError on duplicate or empty name."""
        if not tool.name or not tool.name.strip():
            raise ToolRegistrationError("Tool name cannot be empty")
        if tool.name in self._tools:
            raise ToolRegistrationError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def register_many(self, tools: List[Tool]) -> None:
        """Register multiple tools at once."""
        for tool in tools:
            self.register(tool)

    def unregister(self, name: str) -> bool:
        """Remove a tool by name. Returns True if found and removed."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name. Returns None if not found."""
        return self._tools.get(name)

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get OpenAI function calling schemas for all registered tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def all_tools(self) -> List[Tool]:
        """Get a list of all registered tools."""
        return list(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
