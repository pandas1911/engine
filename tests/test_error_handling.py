"""Error handling tests for the Agent system."""
import asyncio
import pytest

from engine.agent_core import Agent
from engine.config import Config, get_config
from engine.llm_provider import BaseLLMProvider, LLMProviderError, MockLLMProvider
from engine.models import (
    AgentError, AgentResult, AgentState, CollectedChildResult, ErrorCategory, Session, LLMResponse,
)
from engine.registry import SubagentRegistry


class FailingLLMProvider(MockLLMProvider):
    """LLM mock that raises LLMProviderError on specified call count."""

    def __init__(self, fail_on_call=1):
        super().__init__()
        self.fail_on_call = fail_on_call

    async def chat(self, messages, tools=None, **kwargs):
        if self.call_count + 1 == self.fail_on_call:
            raise LLMProviderError(RuntimeError("Simulated LLM failure"))
        return await super().chat(messages, tools, **kwargs)


class RawFailingLLMProvider(MockLLMProvider):
    """LLM mock that raises raw exceptions (for INTERNAL_ERROR classification)."""

    def __init__(self, fail_on_call=1):
        super().__init__()
        self.fail_on_call = fail_on_call

    async def chat(self, messages, tools=None, **kwargs):
        if self.call_count + 1 == self.fail_on_call:
            raise ValueError("Simulated internal error")
        return await super().chat(messages, tools, **kwargs)


class NoSpawnMockProvider(MockLLMProvider):
    """Mock LLM that always returns text, never spawns children."""

    async def chat(self, messages, tools=None, **kwargs):
        return LLMResponse(content="处理完成")


@pytest.fixture
def config():
    return get_config()


@pytest.mark.asyncio
async def test_t1_root_process_tool_calls_error(config):
    """T1: Root agent _process_tool_calls() raises exception -> ERROR state."""
    session = Session(id="test_t1", depth=0)
    agent = Agent(session=session, config=config, llm_provider=FailingLLMProvider(fail_on_call=1))

    result = await agent.run("test task")

    assert agent.state == AgentState.ERROR
    assert "[ERROR]" in result
    assert agent._error_info is not None
    assert agent._error_info.category == ErrorCategory.LLM_ERROR
    assert agent._completion_event.is_set()


@pytest.mark.asyncio
async def test_t2_root_resume_from_children_error(config):
    """T2: Root agent _resume_from_children() raises exception -> ERROR, no hang."""
    session = Session(id="test_t2", depth=0)
    agent = Agent(session=session, config=config, llm_provider=FailingLLMProvider(fail_on_call=2))

    result = await agent.run("test task")

    assert agent.state == AgentState.ERROR
    assert agent._completion_event.is_set()


@pytest.mark.asyncio
async def test_t3_sub_agent_run_error(config):
    """T3: Sub-agent _abort() -> ERROR state, category is INTERNAL_ERROR."""
    session = Session(id="test_t3", depth=0)
    registry = SubagentRegistry()
    agent = Agent(session=session, config=config, registry=registry,
                  llm_provider=MockLLMProvider())
    agent.state_machine.trigger("start")
    await agent._abort(RuntimeError("child crash"))
    assert agent.state == AgentState.ERROR
    assert agent._error_info.category == ErrorCategory.INTERNAL_ERROR


@pytest.mark.asyncio
async def test_t4_sub_agent_resume_error(config):
    """T4: Sub-agent _resume_from_children() raises exception -> ERROR state."""
    session = Session(id="test_t4", depth=1)
    registry = SubagentRegistry()
    agent = Agent(session=session, config=config, registry=registry,
                  llm_provider=FailingLLMProvider(fail_on_call=1),
                  parent_task_id="parent_1")
    agent.state_machine.trigger("start")
    agent.state_machine.trigger("spawn_children")

    child_results = {"child_1": CollectedChildResult(task_description="test", result="done")}
    await agent._resume_from_children(child_results)

    assert agent.state == AgentState.ERROR


@pytest.mark.asyncio
async def test_t5_abort_idempotent(config):
    """T5: _abort() called twice -> no exception, first result preserved."""
    session = Session(id="test_t5", depth=0)
    agent = Agent(session=session, config=config, llm_provider=MockLLMProvider())
    agent.state_machine.trigger("start")

    await agent._abort(RuntimeError("first error"))
    assert agent.state == AgentState.ERROR
    first_result = agent._final_result

    await agent._abort(RuntimeError("second error"))
    assert agent.state == AgentState.ERROR
    assert agent._final_result == first_result  # First result preserved


@pytest.mark.asyncio
async def test_t6_success_path_unchanged(config):
    """T6: Success path -> error=None, success=True (regression test)."""
    session = Session(id="test_t6", depth=0)
    agent = Agent(session=session, config=config, llm_provider=NoSpawnMockProvider())

    result = await agent.run("test task")
    assert agent.state == AgentState.COMPLETED
    assert agent._error_info is None
    assert result is not None


@pytest.mark.asyncio
async def test_t7_llm_provider_error_classification(config):
    """T7: LLM raises LLMProviderError -> error.category=LLM_ERROR."""
    session = Session(id="test_t7", depth=0)
    agent = Agent(session=session, config=config, llm_provider=FailingLLMProvider(fail_on_call=1))

    await agent.run("test task")

    assert agent._error_info is not None
    assert agent._error_info.category == ErrorCategory.LLM_ERROR
    assert "Simulated LLM failure" in agent._error_info.message


@pytest.mark.asyncio
async def test_t8_internal_error_classification(config):
    """T8: Non-LLM exception (ValueError) -> error.category=INTERNAL_ERROR."""
    session = Session(id="test_t8", depth=0)
    agent = Agent(session=session, config=config, llm_provider=RawFailingLLMProvider(fail_on_call=1))

    await agent.run("test task")

    assert agent._error_info is not None
    assert agent._error_info.category == ErrorCategory.INTERNAL_ERROR
    assert "ValueError" in agent._error_info.exception_type
