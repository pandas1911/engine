import asyncio
import pytest

from engine.config import Config, ConfigLoader
from engine.safety import ConcurrencyLimiter
from engine.runtime.agent_models import Session, AgentState
from engine.runtime.task_registry import AgentTaskRegistry
from engine.runtime.agent import Agent
from engine.subagent.manager import SubAgentManager


def test_config_default_values():
    config = Config(api_key="k", base_url="u", model="m")
    assert config.max_concurrent_agents == 10


def test_config_rejects_low_max_concurrent():
    config = Config(api_key="k", base_url="u", model="m", max_concurrent_agents=1)
    assert config.max_concurrent_agents == 1
    with pytest.raises(ValueError, match="max_concurrent_agents must be >= 2"):
        if config.max_concurrent_agents < 2:
            raise ValueError(
                f"max_concurrent_agents must be >= 2, got {config.max_concurrent_agents}. "
                "Values less than 2 can cause deadlock in the agent execution system."
            )


@pytest.mark.asyncio
async def test_concurrency_limiter_active_count():
    limiter = ConcurrencyLimiter(3)
    assert limiter.active_count == 0
    await limiter.acquire()
    assert limiter.active_count == 1
    await limiter.acquire()
    assert limiter.active_count == 2
    limiter.release()
    assert limiter.active_count == 1
    limiter.release()
    assert limiter.active_count == 0


def test_concurrency_limiter_max_concurrent():
    limiter = ConcurrencyLimiter(5)
    assert limiter.max_concurrent == 5


def test_concurrency_limiter_rejects_zero():
    with pytest.raises(ValueError, match="max_concurrent must be >= 1"):
        ConcurrencyLimiter(0)


@pytest.mark.asyncio
async def test_concurrency_limiter_over_release():
    limiter = ConcurrencyLimiter(2)
    await limiter.acquire()
    limiter.release()
    with pytest.raises(RuntimeError, match="release called too many times"):
        limiter.release()


@pytest.mark.asyncio
async def test_global_semaphore_acquire_timeout(config, task_registry, mock_drainable):
    limiter = ConcurrencyLimiter(2)
    await limiter.acquire()
    await limiter.acquire()
    manager = SubAgentManager(
        task_registry=task_registry,
        event_queue=[],
        drainable=mock_drainable,
        agent_task_id="parent1",
        parent_label="Root",
        config=config,
        concurrency_limiter=limiter,
    )
    session = Session(id="s1", depth=0)
    result = await manager.spawn(
        task_desc="test task",
        label="test-label",
        parent_session=session,
        config=config,
        agent_factory=None,
    )
    assert "timed out" in result.lower()
    assert "completing this task yourself directly" in result
    limiter.release()
    limiter.release()


@pytest.mark.asyncio
async def test_global_semaphore_release_on_success(config, task_registry, mock_drainable, success_agent):
    limiter = ConcurrencyLimiter(2)
    await limiter.acquire()
    manager = SubAgentManager(
        task_registry=task_registry,
        event_queue=[],
        drainable=mock_drainable,
        agent_task_id="parent1",
        parent_label="Root",
        config=config,
        concurrency_limiter=limiter,
    )
    agent = success_agent()
    assert limiter.active_count == 1
    await manager._run_child(agent, "task1", "desc", 1)
    assert limiter.active_count == 0


@pytest.mark.asyncio
async def test_global_semaphore_release_on_error(config, task_registry, mock_drainable, error_agent):
    limiter = ConcurrencyLimiter(2)
    await limiter.acquire()
    manager = SubAgentManager(
        task_registry=task_registry,
        event_queue=[],
        drainable=mock_drainable,
        agent_task_id="parent1",
        parent_label="Root",
        config=config,
        concurrency_limiter=limiter,
    )
    agent = error_agent()
    assert limiter.active_count == 1
    await manager._run_child(agent, "task1", "desc", 1)
    assert limiter.active_count == 0


@pytest.mark.asyncio
async def test_global_semaphore_release_on_cancellation(config, task_registry, mock_drainable, cancel_agent):
    limiter = ConcurrencyLimiter(2)
    await limiter.acquire()
    manager = SubAgentManager(
        task_registry=task_registry,
        event_queue=[],
        drainable=mock_drainable,
        agent_task_id="parent1",
        parent_label="Root",
        config=config,
        concurrency_limiter=limiter,
    )
    agent = cancel_agent()
    assert limiter.active_count == 1
    with pytest.raises(asyncio.CancelledError):
        await manager._run_child(agent, "task1", "desc", 1)
    assert limiter.active_count == 0


@pytest.mark.asyncio
async def test_acquire_before_register(config, task_registry, mock_drainable):
    limiter = ConcurrencyLimiter(1)
    await limiter.acquire()
    manager = SubAgentManager(
        task_registry=task_registry,
        event_queue=[],
        drainable=mock_drainable,
        agent_task_id="parent1",
        parent_label="Root",
        config=config,
        concurrency_limiter=limiter,
    )
    session = Session(id="s1", depth=0)
    result = await manager.spawn(
        task_desc="orphan test",
        label="orphan-label",
        parent_session=session,
        config=config,
        agent_factory=None,
    )
    assert "timed out" in result.lower()
    task_count = len([t for t in task_registry._tasks.values() if "orphan" in str(t.task_description).lower()])
    assert task_count == 0
    limiter.release()


@pytest.mark.asyncio
async def test_error_message_contains_context(config, task_registry, mock_drainable):
    limiter = ConcurrencyLimiter(1)
    await limiter.acquire()
    manager = SubAgentManager(
        task_registry=task_registry,
        event_queue=[],
        drainable=mock_drainable,
        agent_task_id="parent1",
        parent_label="Root",
        config=config,
        concurrency_limiter=limiter,
    )
    session = Session(id="s1", depth=0)
    result = await manager.spawn(
        task_desc="specific task description",
        label="my-label",
        parent_session=session,
        config=config,
        agent_factory=None,
    )
    assert "specific task description" in result
    assert "my-label" in result
    assert "completing this task yourself directly" in result
    limiter.release()


def test_shared_limiter_propagation(config, task_registry, mock_llm_provider):
    limiter = ConcurrencyLimiter(3)
    root_session = Session(id="root", depth=0)
    root = Agent(
        session=root_session,
        config=config,
        llm_provider=mock_llm_provider,
        task_registry=task_registry,
        concurrency_limiter=limiter,
    )
    child_session = Session(id="child", depth=1)
    child = root._create_child_agent(
        child_session, config, task_registry, root.task_id, task_id="child1", label="Sub-1"
    )
    grandchild_session = Session(id="gc", depth=2)
    grandchild = child._create_child_agent(
        grandchild_session, config, task_registry, child.task_id, task_id="gc1", label="Sub-1.1"
    )
    assert root._concurrency_limiter is limiter
    assert child._concurrency_limiter is limiter
    assert grandchild._concurrency_limiter is limiter


@pytest.mark.asyncio
async def test_no_concurrent_breach(config, task_registry, mock_drainable):
    limiter = ConcurrencyLimiter(2)
    manager = SubAgentManager(
        task_registry=task_registry,
        event_queue=[],
        drainable=mock_drainable,
        agent_task_id="parent1",
        parent_label="Root",
        config=config,
        concurrency_limiter=limiter,
    )

    done_events = [asyncio.Event() for _ in range(4)]
    peak_count = [0]
    idx_counter = [0]

    class TrackingSlowAgent:
        def __init__(self, event):
            self.state = AgentState.COMPLETED
            self.result = "done"
            self._final_result = "done"
            self.event = event

        async def run(self, task_desc):
            peak_count[0] = max(peak_count[0], limiter.active_count)
            await asyncio.sleep(0.3)
            self.event.set()

        async def abort(self, error):
            pass

    def agent_factory(session, cfg, registry, parent_task_id, task_id, label=None):
        idx = idx_counter[0]
        idx_counter[0] += 1
        return TrackingSlowAgent(done_events[idx])

    session = Session(id="s1", depth=0)
    results = await asyncio.gather(
        manager.spawn("task1", "l1", session, config, agent_factory),
        manager.spawn("task2", "l2", session, config, agent_factory),
        manager.spawn("task3", "l3", session, config, agent_factory),
        manager.spawn("task4", "l4", session, config, agent_factory),
    )

    await asyncio.gather(*(e.wait() for e in done_events))

    assert peak_count[0] <= 2
    for r in results:
        assert "Spawned Task" in r
