"""Extensible tool system for the agent framework.

Provides Tool ABC, FunctionTool factory, and ToolRegistry for managing tools.

Usage:
    from engine.tools import Tool, FunctionTool, ToolRegistry
    
    # Define a simple tool with a function
    my_tool = FunctionTool(
        name="my_tool",
        description="Does something useful",
        parameters={"type": "object", "properties": {...}},
        fn=my_function
    )
    
    # Register with the agent's tool registry
    registry.register(my_tool)
"""

from engine.tools.base import Tool, ToolRegistry, ToolRegistrationError, FunctionTool

__all__ = ["Tool", "ToolRegistry", "ToolRegistrationError", "FunctionTool"]
