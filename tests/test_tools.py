"""Comprehensive unit tests for Tool ABC, FunctionTool, ToolRegistry, and Agent integration.

This module tests:
- Tool ABC: Abstract behavior and required attributes
- FunctionTool: Sync/async wrapping, error handling, attribute storage
- ToolRegistry: Registration, lookup, schemas, duplicates
- Integration: Agent with ToolRegistry and SpawnTool injection
"""

import pytest
import asyncio
from typing import Any, Dict

from src.tools.base import Tool, FunctionTool, ToolRegistry, ToolRegistrationError
from src.tools.builtin.spawn import SpawnTool
from src.agent_core import Agent
from src.models import Session
from src.config import Config


def make_config(**kwargs) -> Config:
    """Helper to create Config with defaults for testing."""
    defaults = {
        'openai_api_key': 'test-key',
        'openai_base_url': 'http://test',
        'openai_model': 'test-model'
    }
    defaults.update(kwargs)
    return Config(**defaults)


class TestToolABC:
    """Tests for Tool abstract base class."""

    def test_tool_is_abstract(self):
        """Verify Tool cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            Tool()
        assert "abstract" in str(exc_info.value).lower() or "cannot instantiate" in str(exc_info.value).lower()

    def test_tool_has_required_attributes(self):
        """Verify Tool defines required class attributes."""
        # Tool is abstract, so we create a concrete subclass to test
        class ConcreteTool(Tool):
            name = "test_tool"
            description = "A test tool"
            parameters = {"type": "object", "properties": {}}

            async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
                return "test result"

        tool = ConcreteTool()
        assert hasattr(tool, 'name')
        assert hasattr(tool, 'description')
        assert hasattr(tool, 'parameters')
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.parameters == {"type": "object", "properties": {}}


class TestFunctionTool:
    """Tests for FunctionTool wrapper."""

    def test_wraps_async_function(self):
        """Test async function wrapping and execution."""
        async def async_fn(args: Dict[str, Any], ctx: Dict[str, Any]) -> str:
            return 'async: ' + args.get('x', '')

        tool = FunctionTool(
            name='async_test',
            description='test async function',
            parameters={'type': 'object', 'properties': {'x': {'type': 'string'}}},
            fn=async_fn
        )
        result = asyncio.run(tool.execute({'x': 'world'}, {}))
        assert result == 'async: world'

    def test_wraps_sync_function(self):
        """Test sync function wrapping and execution."""
        def sync_fn(args: Dict[str, Any], ctx: Dict[str, Any]) -> str:
            return 'sync: ' + args.get('x', '')

        tool = FunctionTool(
            name='sync_test',
            description='test sync function',
            parameters={'type': 'object', 'properties': {'x': {'type': 'string'}}},
            fn=sync_fn
        )
        result = asyncio.run(tool.execute({'x': 'hello'}, {}))
        assert result == 'sync: hello'

    def test_result_forced_to_string(self):
        """Test that non-string return values are converted to string."""
        def returns_int(args: Dict[str, Any], ctx: Dict[str, Any]) -> int:
            return 42

        def returns_dict(args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict:
            return {'key': 'value'}

        def returns_list(args: Dict[str, Any], ctx: Dict[str, Any]) -> list:
            return [1, 2, 3]

        tool_int = FunctionTool(
            name='int_test',
            description='returns int',
            parameters={},
            fn=returns_int
        )
        tool_dict = FunctionTool(
            name='dict_test',
            description='returns dict',
            parameters={},
            fn=returns_dict
        )
        tool_list = FunctionTool(
            name='list_test',
            description='returns list',
            parameters={},
            fn=returns_list
        )

        assert asyncio.run(tool_int.execute({}, {})) == '42'
        assert asyncio.run(tool_dict.execute({}, {})) == "{'key': 'value'}"
        assert asyncio.run(tool_list.execute({}, {})) == '[1, 2, 3]'

    def test_rejects_empty_name(self):
        """Test that empty name raises ToolRegistrationError."""
        with pytest.raises(ToolRegistrationError) as exc_info:
            FunctionTool(
                name='',
                description='test',
                parameters={},
                fn=lambda a, c: ''
            )
        assert 'empty' in str(exc_info.value).lower() or 'name' in str(exc_info.value).lower()

    def test_rejects_whitespace_name(self):
        """Test that whitespace-only name raises ToolRegistrationError."""
        with pytest.raises(ToolRegistrationError) as exc_info:
            FunctionTool(
                name='   ',
                description='test',
                parameters={},
                fn=lambda a, c: ''
            )
        assert 'empty' in str(exc_info.value).lower() or 'name' in str(exc_info.value).lower()

    def test_stores_attributes(self):
        """Test that name, description, parameters are stored correctly."""
        params = {
            'type': 'object',
            'properties': {
                'input': {'type': 'string'}
            },
            'required': ['input']
        }

        tool = FunctionTool(
            name='my_tool',
            description='Does something useful',
            parameters=params,
            fn=lambda a, c: 'done'
        )

        assert tool.name == 'my_tool'
        assert tool.description == 'Does something useful'
        assert tool.parameters == params


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_single_tool(self):
        """Test registering a single tool."""
        registry = ToolRegistry()
        tool = FunctionTool(
            name='test_tool',
            description='test',
            parameters={},
            fn=lambda a, c: 'result'
        )

        registry.register(tool)
        assert len(registry) == 1
        assert 'test_tool' in registry

    def test_register_many_tools(self):
        """Test batch registration with register_many."""
        registry = ToolRegistry()
        tools = [
            FunctionTool(name=f'tool_{i}', description=f'tool {i}', parameters={}, fn=lambda a, c: str(i))
            for i in range(3)
        ]

        registry.register_many(tools)
        assert len(registry) == 3
        for i in range(3):
            assert f'tool_{i}' in registry

    def test_get_existing_tool(self):
        """Test getting a tool by name."""
        registry = ToolRegistry()
        tool = FunctionTool(
            name='my_tool',
            description='test',
            parameters={},
            fn=lambda a, c: 'result'
        )

        registry.register(tool)
        retrieved = registry.get('my_tool')
        assert retrieved is not None
        assert retrieved.name == 'my_tool'

    def test_get_nonexistent_tool(self):
        """Test that get returns None for missing tool."""
        registry = ToolRegistry()
        result = registry.get('nonexistent')
        assert result is None

    def test_rejects_duplicate_name(self):
        """Test that duplicate registration raises ToolRegistrationError."""
        registry = ToolRegistry()
        tool1 = FunctionTool(
            name='dup_tool',
            description='first',
            parameters={},
            fn=lambda a, c: '1'
        )
        tool2 = FunctionTool(
            name='dup_tool',
            description='second',
            parameters={},
            fn=lambda a, c: '2'
        )

        registry.register(tool1)
        with pytest.raises(ToolRegistrationError) as exc_info:
            registry.register(tool2)
        assert 'already registered' in str(exc_info.value) or 'dup_tool' in str(exc_info.value)

    def test_rejects_empty_name_tool(self):
        """Test that registering tool with empty name raises ToolRegistrationError."""
        registry = ToolRegistry()

        # Create a mock tool with empty name
        class EmptyNameTool(Tool):
            name = ""
            description = "test"
            parameters = {}

            async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
                return ""

        with pytest.raises(ToolRegistrationError) as exc_info:
            registry.register(EmptyNameTool())
        assert 'empty' in str(exc_info.value).lower() or 'name' in str(exc_info.value).lower()

    def test_get_schemas_format(self):
        """Test that schema output matches OpenAI function calling format."""
        registry = ToolRegistry()
        tool = FunctionTool(
            name='test_fn',
            description='A test function',
            parameters={
                'type': 'object',
                'properties': {'arg1': {'type': 'string'}},
                'required': ['arg1']
            },
            fn=lambda a, c: 'result'
        )

        registry.register(tool)
        schemas = registry.get_schemas()

        assert len(schemas) == 1
        schema = schemas[0]
        assert schema['type'] == 'function'
        assert 'function' in schema
        assert schema['function']['name'] == 'test_fn'
        assert schema['function']['description'] == 'A test function'
        assert 'parameters' in schema['function']

    def test_empty_registry(self):
        """Test empty registry behavior."""
        registry = ToolRegistry()

        assert len(registry) == 0
        assert registry.get_schemas() == []
        assert registry.all_tools() == []
        assert 'anything' not in registry
        assert registry.get('anything') is None

    def test_len_and_contains(self):
        """Test __len__ and __contains__ work."""
        registry = ToolRegistry()

        assert len(registry) == 0
        assert 'tool1' not in registry

        tool = FunctionTool(
            name='tool1',
            description='test',
            parameters={},
            fn=lambda a, c: 'result'
        )
        registry.register(tool)

        assert len(registry) == 1
        assert 'tool1' in registry
        assert 'tool2' not in registry

    def test_all_tools(self):
        """Test all_tools returns all registered tools."""
        registry = ToolRegistry()
        tools = [
            FunctionTool(name='tool_a', description='a', parameters={}, fn=lambda a, c: 'a'),
            FunctionTool(name='tool_b', description='b', parameters={}, fn=lambda a, c: 'b'),
        ]

        registry.register_many(tools)
        all_tools = registry.all_tools()

        assert len(all_tools) == 2
        names = [t.name for t in all_tools]
        assert 'tool_a' in names
        assert 'tool_b' in names


class TestIntegration:
    """Tests for Agent + ToolRegistry integration."""

    def test_agent_with_tool_registry(self):
        """Test Agent accepts tool_registry parameter."""
        reg = ToolRegistry()
        reg.register(FunctionTool(
            name='custom',
            description='test',
            parameters={},
            fn=lambda a, c: 'hi'
        ))

        agent = Agent(
            Session(id='s1', depth=0),
            make_config(),
            tool_registry=reg
        )

        assert agent._tool_registry.get('custom') is not None

    def test_agent_backward_compat_tools_list(self):
        """Test Agent accepts old tools list parameter."""
        tools = [
            FunctionTool(name='tool1', description='t1', parameters={}, fn=lambda a, c: '1'),
            FunctionTool(name='tool2', description='t2', parameters={}, fn=lambda a, c: '2'),
        ]

        agent = Agent(
            Session(id='s1', depth=0),
            make_config(),
            tools=tools
        )

        assert agent._tool_registry.get('tool1') is not None
        assert agent._tool_registry.get('tool2') is not None

    def test_spawn_tool_auto_injected(self):
        """Test SpawnTool is conditionally injected."""
        cfg = make_config(max_depth=3)
        agent = Agent(
            Session(id='s2', depth=0),
            cfg
        )

        spawn_tool = agent._tool_registry.get('spawn')
        assert spawn_tool is not None
        assert isinstance(spawn_tool, SpawnTool)

    def test_spawn_tool_not_injected_at_max_depth(self):
        """Test max depth does not inject SpawnTool."""
        cfg = make_config(max_depth=3)
        agent = Agent(
            Session(id='s3', depth=3),
            cfg
        )

        spawn_tool = agent._tool_registry.get('spawn')
        assert spawn_tool is None

    def test_mixed_tools_in_registry(self):
        """Test both class-based tools and FunctionTool work together."""

        # Class-based tool
        class MyTool(Tool):
            name = "my_class_tool"
            description = "A class-based tool"
            parameters = {"type": "object", "properties": {}}

            async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
                return "class tool result"

        # Function-based tool
        func_tool = FunctionTool(
            name="my_func_tool",
            description="A function-based tool",
            parameters={},
            fn=lambda a, c: "func tool result"
        )

        reg = ToolRegistry()
        reg.register(MyTool())
        reg.register(func_tool)

        assert reg.get('my_class_tool') is not None
        assert reg.get('my_func_tool') is not None

        # Verify both can be executed
        class_tool = reg.get('my_class_tool')
        func_tool_retrieved = reg.get('my_func_tool')

        class_result = asyncio.run(class_tool.execute({}, {}))
        func_result = asyncio.run(func_tool_retrieved.execute({}, {}))

        assert class_result == "class tool result"
        assert func_result == "func tool result"
