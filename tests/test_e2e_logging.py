"""End-to-end integration test for the logging pipeline with MockLLMProvider."""

import json
import os
import pytest


@pytest.fixture(autouse=True)
def reset_logger():
    """Reset logger singleton between tests."""
    import engine.logger as logger_mod
    logger_mod._logger = None
    yield
    logger_mod._logger = None


@pytest.mark.asyncio
async def test_delegate_creates_log_file(tmp_path):
    """Verify that delegate() creates a .jsonl log file."""
    from engine.agent_core import Agent
    from engine.config import Config
    from engine.llm_provider import MockLLMProvider
    from engine.logger import init_logger, stop_logger
    from engine.models import Session
    from engine.subagent.registry import SubagentRegistry

    config = Config(
        api_key="test",
        base_url="http://localhost",
        model="test",
        log_dir=str(tmp_path),
    )

    init_logger(log_dir=str(tmp_path))

    session = Session(id="test_root", depth=0)
    session.add_message("system", "You are a test agent.")

    registry = SubagentRegistry()
    mock_llm = MockLLMProvider()

    agent = Agent(
        session=session,
        config=config,
        registry=registry,
        llm_provider=mock_llm,
    )

    await registry.register(
        task_id=agent.task_id,
        session_id=session.id,
        description="test task",
        parent_agent=None,
        agent=agent,
        depth=0,
    )

    await agent.run("Test task")

    if agent.state.value != "completed":
        await agent._completion_event.wait()

    await stop_logger()

    # Verify log file exists
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) >= 1, "No .jsonl files found in {}".format(tmp_path)

    # Verify log file has content
    with open(files[0]) as f:
        lines = f.readlines()
    assert len(lines) > 0, "Log file is empty"

    # Verify each line is valid JSON
    for line in lines:
        entry = json.loads(line)
        assert "timestamp" in entry
        assert "level" in entry


@pytest.mark.asyncio
async def test_log_contains_lifecycle_events(tmp_path):
    """Verify log captures agent_init, agent_run_start, and state_change events."""
    from engine.agent_core import Agent
    from engine.config import Config
    from engine.llm_provider import MockLLMProvider
    from engine.logger import init_logger, stop_logger
    from engine.models import Session
    from engine.subagent.registry import SubagentRegistry

    config = Config(
        api_key="test", base_url="http://localhost", model="test"
    )

    init_logger(log_dir=str(tmp_path))

    session = Session(id="test_lifecycle", depth=0)
    session.add_message("system", "Test")
    registry = SubagentRegistry()
    mock_llm = MockLLMProvider()

    agent = Agent(
        session=session, config=config, registry=registry, llm_provider=mock_llm
    )

    await registry.register(
        task_id=agent.task_id,
        session_id=session.id,
        description="test",
        parent_agent=None,
        agent=agent,
        depth=0,
    )

    await agent.run("Test task")

    if agent.state.value != "completed":
        await agent._completion_event.wait()

    await stop_logger()

    files = list(tmp_path.glob("*.jsonl"))
    with open(files[0]) as f:
        entries = [json.loads(line) for line in f]

    event_types = [e["event_type"] for e in entries]

    # Must have these event types
    assert "agent_init" in event_types, "Missing agent_init event"
    assert "agent_run_start" in event_types, "Missing agent_run_start event"
    assert "state_change" in event_types, "Missing state_change event"


@pytest.mark.asyncio
async def test_custom_log_directory_works(tmp_path):
    """Verify custom log_dir from Config is respected."""
    from engine.agent_core import Agent
    from engine.config import Config
    from engine.llm_provider import MockLLMProvider
    from engine.logger import init_logger, stop_logger
    from engine.models import Session
    from engine.subagent.registry import SubagentRegistry

    custom_dir = str(tmp_path / "my_custom_logs")
    config = Config(
        api_key="test",
        base_url="http://localhost",
        model="test",
        log_dir=custom_dir,
    )

    init_logger(log_dir=config.log_dir)

    session = Session(id="test_custom", depth=0)
    session.add_message("system", "Test")
    registry = SubagentRegistry()
    mock_llm = MockLLMProvider()

    agent = Agent(
        session=session, config=config, registry=registry, llm_provider=mock_llm
    )

    await registry.register(
        task_id=agent.task_id,
        session_id=session.id,
        description="test",
        parent_agent=None,
        agent=agent,
        depth=0,
    )

    await agent.run("Test")

    if agent.state.value != "completed":
        await agent._completion_event.wait()

    await stop_logger()

    assert os.path.isdir(custom_dir), "Custom log directory not created"
    files = [f for f in os.listdir(custom_dir) if f.endswith(".jsonl")]
    assert len(files) >= 1, "No log files in custom directory"


@pytest.mark.asyncio
async def test_root_agent_label_in_logs(tmp_path):
    """Verify root agent has label 'Root' in log entries."""
    from engine.agent_core import Agent
    from engine.config import Config
    from engine.llm_provider import MockLLMProvider
    from engine.logger import init_logger, stop_logger
    from engine.models import Session
    from engine.subagent.registry import SubagentRegistry

    config = Config(
        api_key="test", base_url="http://localhost", model="test"
    )
    init_logger(log_dir=str(tmp_path))

    session = Session(id="test_label", depth=0)
    session.add_message("system", "Test")
    registry = SubagentRegistry()
    mock_llm = MockLLMProvider()

    agent = Agent(
        session=session, config=config, registry=registry, llm_provider=mock_llm
    )
    assert agent.label == "Root", "Root agent label should be 'Root'"

    await registry.register(
        task_id=agent.task_id,
        session_id=session.id,
        description="test",
        parent_agent=None,
        agent=agent,
        depth=0,
    )

    await agent.run("Test")

    if agent.state.value != "completed":
        await agent._completion_event.wait()

    await stop_logger()

    files = list(tmp_path.glob("*.jsonl"))
    with open(files[0]) as f:
        entries = [json.loads(line) for line in f]

    root_entries = [e for e in entries if e["agent_label"] == "Root"]
    assert len(root_entries) > 0, "No log entries with label 'Root'"
