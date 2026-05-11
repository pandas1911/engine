"""Tests for ChildCompletionNotification model and related data model changes."""

import pytest

from engine.subagent.subagent_models import ChildCompletionNotification
from engine.subagent.events import AgentEvent, ChildCompletionEvent
from engine.runtime.agent_models import QueueEvent


class TestChildCompletionNotification:
    def test_instantiation_with_all_fields(self):
        notif = ChildCompletionNotification(
            task_id="task_abc123",
            label="Sub-1(d:1)",
            task="Analyze the codebase",
            status="completed",
            summary="Found 3 issues in the parser module.",
            session_file="task_abc123.json",
        )
        assert notif.task_id == "task_abc123"
        assert notif.label == "Sub-1(d:1)"
        assert notif.task == "Analyze the codebase"
        assert notif.status == "completed"
        assert notif.summary == "Found 3 issues in the parser module."
        assert notif.session_file == "task_abc123.json"

    def test_to_prompt_format(self):
        notif = ChildCompletionNotification(
            task_id="task_abc123",
            label="Sub-1(d:1)",
            task="Analyze the codebase",
            status="completed",
            summary="Found 3 issues in the parser module.",
            session_file="task_abc123.json",
        )
        prompt = notif.to_prompt()
        assert "[Child Agent Report] Sub-1(d:1) (task_abc123) has completed:" in prompt
        assert "- Status: completed" in prompt
        assert "- Task: Analyze the codebase" in prompt
        assert "- Summary: Found 3 issues in the parser module." in prompt

    def test_to_prompt_error_status(self):
        notif = ChildCompletionNotification(
            task_id="task_err",
            label="Sub-2(d:2)",
            task="Run tests",
            status="error",
            summary="Tests failed with exit code 1.",
            session_file="task_err.json",
        )
        prompt = notif.to_prompt()
        assert "- Status: error" in prompt


class TestChildCompletionEvent:
    def test_carries_single_notification(self):
        notif = ChildCompletionNotification(
            task_id="task_abc123",
            label="Sub-1(d:1)",
            task="Analyze the codebase",
            status="completed",
            summary="Done.",
            session_file="task_abc123.json",
        )
        event = ChildCompletionEvent(notification=notif)
        assert isinstance(event, AgentEvent)
        assert event.notification is notif
        assert event.notification.task_id == "task_abc123"


class TestCollectedChildResultRemoved:
    def test_no_longer_importable(self):
        import engine.subagent.subagent_models as models

        assert not hasattr(models, "CollectedChildResult")


class TestQueueEvent:
    def test_uses_child_summary_string(self):
        qe = QueueEvent(
            trigger_task_id="task_child1",
            child_summary="Child task_child1 completed: Done.",
            error=False,
        )
        assert qe.trigger_task_id == "task_child1"
        assert qe.child_summary == "Child task_child1 completed: Done."
        assert qe.error is False

    def test_error_queue_event(self):
        qe = QueueEvent(
            trigger_task_id="task_child_err",
            child_summary="Child task_child_err errored: OOM",
            error=True,
        )
        assert qe.error is True
