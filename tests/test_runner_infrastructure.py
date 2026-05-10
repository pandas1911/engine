"""Tests for Engine, SessionManager, and MessageEvent."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from engine.runtime.agent import MessageEvent
from engine.runtime.agent_models import AgentState
from engine.runner import Engine, SessionManager


class TestMessageEvent:
    def test_message_event_class(self):
        event = MessageEvent("hello")
        assert event.content == "hello"

    def test_message_event_empty(self):
        event = MessageEvent("")
        assert event.content == ""


class TestEngine:
    def setup_method(self):
        Engine.reset()

    def test_engine_singleton(self):
        with patch.object(Engine, "__init__", lambda self, config=None: None):
            e1 = Engine.get()
            e2 = Engine.get()
            assert e1 is e2

    def test_engine_reset(self):
        with patch.object(Engine, "__init__", lambda self, config=None: None):
            e1 = Engine.get()
            Engine.reset()
            e2 = Engine.get()
            assert e1 is not e2


class TestSessionManagerInterject:
    def _make_mgr(self, agent_state):
        with patch("engine.session_store.SessionStore"), \
             patch("engine.runner.Agent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent.state = agent_state
            mock_agent.task_id = "test_task"
            MockAgent.return_value = mock_agent

            infra = MagicMock()
            infra.config = MagicMock()
            infra.config.is_tool_enabled.return_value = True
            infra.config.user_timezone = None

            mgr = SessionManager(infra=infra)
            mgr.agent = mock_agent
            return mgr

    def test_interject_waiting_direct_run(self):
        mgr = self._make_mgr(AgentState.WAITING_FOR_CHILDREN)
        with patch("asyncio.create_task") as mock_create_task:
            result = mgr.interject("hello")
            assert result == "accepted"
            mock_create_task.assert_called_once()

    def test_interject_running_queues_message(self):
        mgr = self._make_mgr(AgentState.RUNNING)
        result = mgr.interject("hello")
        assert result == "queued"
        assert len(mgr._event_queue) == 1
        assert isinstance(mgr._event_queue[0], MessageEvent)
        assert mgr._event_queue[0].content == "hello"

    def test_interject_completed_rejected(self):
        mgr = self._make_mgr(AgentState.COMPLETED)
        result = mgr.interject("test")
        assert result == "rejected"

    def test_interject_error_rejected(self):
        mgr = self._make_mgr(AgentState.ERROR)
        result = mgr.interject("test")
        assert result == "rejected"
