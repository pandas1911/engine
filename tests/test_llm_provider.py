"""Unit tests for universal LLMProvider (TDD red phase).

These tests define the expected interface of the new LLMProvider.
They will FAIL until the new implementation is created.
"""

import sys
import pytest
from typing import Dict, List, Any, Optional
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, '/Users/sys/Desktop/engine')

from src.llm_provider import BaseLLMProvider, LLMProvider, MockLLMProvider
from src.config import Config
from src.models import LLMResponse, ToolCall


@pytest.fixture
def mock_config():
    """Provide a mock Config for testing."""
    return Config(
        api_key='test-key',
        base_url='http://test.com/v1',
        model='test-model'
    )


@pytest.fixture
def mock_config_no_strip():
    """Provide a mock Config with strip_thinking=False."""
    return Config(
        api_key='test-key',
        base_url='http://test.com/v1',
        model='test-model',
        strip_thinking=False
    )


class TestProviderInstantiation:
    """Test provider instantiation and basic properties."""

    def test_provider_instantiation(self, mock_config):
        """LLMProvider can be instantiated with mock Config."""
        provider = LLMProvider(mock_config)

        assert provider.model == 'test-model'
        assert provider.strip_thinking is True


class TestChatFunctionality:
    """Test chat() method behavior."""

    @pytest.mark.asyncio
    async def test_chat_passes_standard_messages(self, mock_config):
        """Mock AsyncOpenAI client. Call chat() with standard messages.

        Verify client.chat.completions.create was called with messages EXACTLY
        as passed (no conversion).
        """
        messages = [{"role": "user", "content": "hello"}]
        tools = []

        with patch('src.llm_provider.AsyncOpenAI') as mock_client_class:
            mock_client = MagicMock()
            mock_create = AsyncMock()
            mock_client.chat.completions.create = mock_create
            mock_client_class.return_value = mock_client

            # Create mock response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "test response"
            mock_response.choices[0].message.tool_calls = None
            mock_create.return_value = mock_response

            provider = LLMProvider(mock_config)
            await provider.chat(messages, tools)

            # Verify messages passed EXACTLY as-is (no conversion)
            call_args = mock_create.call_args[1]
            assert call_args["messages"] == messages

    @pytest.mark.asyncio
    async def test_chat_with_system_message(self, mock_config):
        """Pass messages with system role.

        Verify system message is passed through UNCHANGED (no MiniMax-style conversion).
        """
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hi"}
        ]
        tools = []

        with patch('src.llm_provider.AsyncOpenAI') as mock_client_class:
            mock_client = MagicMock()
            mock_create = AsyncMock()
            mock_client.chat.completions.create = mock_create
            mock_client_class.return_value = mock_client

            # Create mock response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "test response"
            mock_response.choices[0].message.tool_calls = None
            mock_create.return_value = mock_response

            provider = LLMProvider(mock_config)
            await provider.chat(messages, tools)

            # Verify system message passed through unchanged
            call_args = mock_create.call_args[1]
            assert call_args["messages"] == messages
            assert call_args["messages"][0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_chat_with_tool_calls_response(self, mock_config):
        """Mock response to include tool_calls.

        Verify they are parsed into ToolCall objects with correct name,
        arguments (dict), call_id.
        """
        messages = [{"role": "user", "content": "hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool"}}]

        with patch('src.llm_provider.AsyncOpenAI') as mock_client_class:
            mock_client = MagicMock()
            mock_create = AsyncMock()
            mock_client.chat.completions.create = mock_create
            mock_client_class.return_value = mock_client

            # Create mock response with tool_calls
            mock_tool_call = MagicMock()
            mock_tool_call.function.name = "test_tool"
            mock_tool_call.function.arguments = '{"arg1": "value1"}'
            mock_tool_call.id = "call_123"

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = None
            mock_response.choices[0].message.tool_calls = [mock_tool_call]
            mock_create.return_value = mock_response

            provider = LLMProvider(mock_config)
            result = await provider.chat(messages, tools)

            # Verify tool calls parsed correctly
            assert isinstance(result, LLMResponse)
            assert len(result.tool_calls) == 1
            assert isinstance(result.tool_calls[0], ToolCall)
            assert result.tool_calls[0].name == "test_tool"
            assert result.tool_calls[0].arguments == {"arg1": "value1"}
            assert result.tool_calls[0].call_id == "call_123"

    @pytest.mark.asyncio
    async def test_chat_returns_content(self, mock_config):
        """Mock response with simple text content.

        Verify LLMResponse returned with correct content.
        """
        messages = [{"role": "user", "content": "hello"}]
        tools = []

        with patch('src.llm_provider.AsyncOpenAI') as mock_client_class:
            mock_client = MagicMock()
            mock_create = AsyncMock()
            mock_client.chat.completions.create = mock_create
            mock_client_class.return_value = mock_client

            # Create mock response with content
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Hello, how can I help you?"
            mock_response.choices[0].message.tool_calls = None
            mock_create.return_value = mock_response

            provider = LLMProvider(mock_config)
            result = await provider.chat(messages, tools)

            # Verify response content
            assert isinstance(result, LLMResponse)
            assert result.content == "Hello, how can I help you?"
            assert result.tool_calls == []


class TestStripThinking:
    """Test _strip_thinking() method."""

    def test_strip_thinking_removes_tags(self, mock_config):
        """Test _strip_thinking() with content containing think tags.

        Verify tags removed, visible content preserved.
        """
        provider = LLMProvider(mock_config)

        # Build content with string concatenation to avoid XML parsing issues
        tag_open = "<" + "think" + ">"
        tag_close = "<" + "/think" + ">"
        content = "Hello " + tag_open + "some thinking here" + tag_close + " world"
        result = provider._strip_thinking(content)

        assert tag_open not in result
        assert tag_close not in result
        assert "some thinking here" not in result
        assert "Hello" in result
        assert "world" in result

    def test_strip_thinking_preserves_normal_content(self, mock_config):
        """Test _strip_thinking() with normal text.

        Verify content is NOT filtered (old skip_patterns logic is gone).
        """
        provider = LLMProvider(mock_config)

        # These should NOT be filtered anymore
        contents = [
            "I should do this",
            "Let me think about it",
            "I need to complete this task",
            "I will help you",
            "I can do that"
        ]

        for content in contents:
            result = provider._strip_thinking(content)
            assert result == content

    def test_strip_thinking_disabled(self, mock_config_no_strip):
        """Create provider with strip_thinking=False.

        Test that _strip_thinking() returns content unchanged (including think tags).
        """
        provider = LLMProvider(mock_config_no_strip)

        # Build content with string concatenation to avoid XML parsing issues
        tag_open = "<" + "think" + ">"
        tag_close = "<" + "/think" + ">"
        content = "Hello " + tag_open + "some thinking" + tag_close + " world"
        result = provider._strip_thinking(content)

        # Content should be unchanged when strip_thinking is False
        assert result == content
        assert tag_open in result


class TestEdgeCases:
    """Test edge cases and special handling."""

    def test_none_content_handling(self, mock_config):
        """Test _strip_thinking(None) returns "" and _strip_thinking("") returns ""."""
        provider = LLMProvider(mock_config)

        assert provider._strip_thinking(None) == ""
        assert provider._strip_thinking("") == ""

    @pytest.mark.asyncio
    async def test_stream_chat_not_implemented(self, mock_config):
        """Verify stream_chat() raises NotImplementedError."""
        provider = LLMProvider(mock_config)

        messages = [{"role": "user", "content": "hello"}]
        tools = []

        with pytest.raises(NotImplementedError):
            await provider.stream_chat(messages, tools)


class TestMockProvider:
    """Test MockLLMProvider still works."""

    @pytest.mark.asyncio
    async def test_mock_provider_still_works(self):
        """Verify MockLLMProvider still works: inherits from BaseLLMProvider,
        chat() returns LLMResponse, call_count increments.
        """
        # Verify inheritance
        assert issubclass(MockLLMProvider, BaseLLMProvider)

        provider = MockLLMProvider()
        messages = [{"role": "user", "content": "hello"}]
        tools = []

        # Verify call_count starts at 0
        assert provider.call_count == 0

        # Verify chat() returns LLMResponse
        result = await provider.chat(messages, tools)
        assert isinstance(result, LLMResponse)

        # Verify call_count increments
        assert provider.call_count == 1

        # Verify it increments again on subsequent calls
        await provider.chat(messages, tools)
        assert provider.call_count == 2
