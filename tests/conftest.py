import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest

from engine.config import Config, ConfigProvider, ConfigLoader
from engine.safety import ConcurrencyLimiter
from engine.runtime.agent_models import Session, AgentState
from engine.runtime.task_registry import AgentTaskRegistry


@pytest.fixture
def config():
    return Config(
        api_key="test-key",
        base_url="http://test.example.com",
        model="test-model",
        max_concurrent_agents=3,
        spawn_timeout=0.5,
    )


@pytest.fixture
def limiter():
    return ConcurrencyLimiter(3)


@pytest.fixture
def task_registry():
    return AgentTaskRegistry()


@pytest.fixture
def mock_llm_provider():
    provider = AsyncMock()
    response = MagicMock()
    response.has_tool_calls.return_value = False
    response.content = "mock response"
    response.tool_calls = []
    provider.chat.return_value = response
    return provider


@pytest.fixture
def mock_drainable():
    drainable = MagicMock()
    drainable.state = AgentState.RUNNING
    return drainable


class MockConfigProvider(ConfigProvider):
    def __init__(self, values=None):
        self.values = values or {
            "LLM_API_KEY": "test-key",
            "LLM_BASE_URL": "http://test.example.com",
            "LLM_MODEL": "test-model",
        }

    def get(self, key):
        return self.values.get(key)


@pytest.fixture
def mock_config_provider():
    return MockConfigProvider()


class SuccessAgent:
    def __init__(self):
        self.state = AgentState.COMPLETED
        self.result = "done"
        self._final_result = "done"

    async def run(self, task_desc):
        pass

    async def abort(self, error):
        pass


class ErrorAgent:
    def __init__(self):
        self.state = AgentState.COMPLETED
        self.result = "done"
        self._final_result = "done"

    async def run(self, task_desc):
        raise RuntimeError("simulated error")

    async def abort(self, error):
        self.state = AgentState.ERROR
        self._final_result = str(error)


class CancelAgent:
    def __init__(self):
        self.state = AgentState.COMPLETED
        self.result = "done"
        self._final_result = "done"

    async def run(self, task_desc):
        raise asyncio.CancelledError("simulated cancellation")

    async def abort(self, error):
        pass


class SlowAgent:
    def __init__(self, delay=0.3):
        self.state = AgentState.COMPLETED
        self.result = "done"
        self._final_result = "done"
        self.delay = delay

    async def run(self, task_desc):
        await asyncio.sleep(self.delay)

    async def abort(self, error):
        pass


@pytest.fixture
def success_agent():
    return SuccessAgent


@pytest.fixture
def error_agent():
    return ErrorAgent


@pytest.fixture
def cancel_agent():
    return CancelAgent


@pytest.fixture
def slow_agent():
    return SlowAgent
