"""Tests for SessionStore file persistence layer."""

import json
from pathlib import Path

import pytest

from engine.runtime.agent_models import Session, Message
from engine.subagent.session_store import SessionStore, ChildSessionInfo


@pytest.fixture
def store(tmp_path):
    """Create a SessionStore with a temporary root directory."""
    return SessionStore(str(tmp_path))


@pytest.fixture
def root_session():
    """Create a root session with a couple of messages."""
    return Session(id="root_001", depth=0, parent_id=None, messages=[
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there"),
    ])


@pytest.fixture
def child_session():
    """Create a child session with messages."""
    return Session(id="child_001", depth=1, parent_id="root_001", messages=[
        Message(role="user", content="Do the thing"),
        Message(role="assistant", content="Done"),
    ])


def _init_store(store, root_session_id="root_001"):
    """Helper: initialize store with a root session directory."""
    return store.create_root(root_session_id)


class TestCreateRoot:
    def test_creates_directory(self, store, tmp_path):
        result = _init_store(store)

        assert result == tmp_path / "root_001"
        assert result.is_dir()

    def test_idempotent(self, store):
        path1 = _init_store(store)
        path2 = _init_store(store)

        assert path1 == path2
        assert path1.is_dir()


class TestSaveChildSession:
    def test_creates_task_json(self, store, child_session):
        _init_store(store)
        store.save_child_session("task_abc123", child_session)

        file_path = store.sessions_dir / "task_abc123.json"
        assert file_path.exists()

    def test_file_contains_full_session(self, store, child_session):
        _init_store(store)
        store.save_child_session("task_abc123", child_session)

        data = json.loads((store.sessions_dir / "task_abc123.json").read_text())

        assert data["id"] == "child_001"
        assert data["depth"] == 1
        assert data["parent_id"] == "root_001"
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "Do the thing"


class TestRoundTrip:
    def test_serialize_deserialize_roundtrip(self, store, child_session):
        _init_store(store)
        store.save_child_session("task_xyz", child_session)

        restored = store.read_child_session("task_xyz")

        assert restored is not None
        assert restored.id == child_session.id
        assert restored.depth == child_session.depth
        assert restored.parent_id == child_session.parent_id
        assert len(restored.messages) == len(child_session.messages)
        for orig, rest in zip(child_session.messages, restored.messages):
            assert orig.role == rest.role
            assert orig.content == rest.content
            assert orig.metadata == rest.metadata

    def test_read_nonexistent_returns_none(self, store):
        _init_store(store)

        assert store.read_child_session("task_nothing") is None


class TestListChildren:
    def test_returns_child_info(self, store, child_session):
        _init_store(store)
        store.save_child_session("task_alpha", child_session)
        store.save_child_session("task_beta", child_session)

        children = store.list_children()

        assert len(children) == 2
        task_ids = {c.task_id for c in children}
        assert task_ids == {"task_alpha", "task_beta"}

    def test_child_info_metadata(self, store, child_session):
        _init_store(store)
        store.save_child_session("task_abc", child_session)

        info = store.list_children()[0]

        assert isinstance(info, ChildSessionInfo)
        assert info.task_id == "task_abc"
        assert info.message_count == 2
        assert info.file_size_bytes > 0
        assert "task_abc.json" in info.file_path

    def test_empty_directory(self, store):
        _init_store(store)

        assert store.list_children() == []

    def test_skips_main_json(self, store, root_session, child_session):
        _init_store(store)
        store.save_main_session(root_session)
        store.save_child_session("task_child1", child_session)

        children = store.list_children()

        assert len(children) == 1
        assert children[0].task_id == "task_child1"


class TestPartialSession:
    def test_partial_session_valid_json(self, store, child_session):
        """Crash scenario: a partial session file with valid JSON is readable."""
        _init_store(store)

        child_session.add_message("user", "Partial work")
        store.save_child_session("task_partial", child_session)

        restored = store.read_child_session("task_partial")

        assert restored is not None
        assert len(restored.messages) == 3

    def test_corrupted_file_returns_none(self, store):
        """A file with invalid JSON returns None gracefully."""
        _init_store(store)
        bad_file = store.sessions_dir / "task_corrupt.json"
        bad_file.write_text("{invalid json content", encoding="utf-8")

        assert store.read_child_session("task_corrupt") is None

    def test_corrupted_file_list_children(self, store):
        """Corrupted files get message_count=-1 in list_children."""
        _init_store(store)
        bad_file = store.sessions_dir / "task_bad.json"
        bad_file.write_text("not json at all", encoding="utf-8")

        children = store.list_children()

        assert len(children) == 1
        assert children[0].message_count == -1


class TestAppendMessage:
    def test_persists_and_updates_in_memory(self, store):
        _init_store(store)
        session = Session(id="append_001", depth=1, parent_id="root_001")

        store.append_message("task_append", session, "user", "First msg")
        store.append_message("task_append", session, "assistant", "Reply")

        assert len(session.messages) == 2

        restored = store.read_child_session("task_append")
        assert restored is not None
        assert len(restored.messages) == 2
        assert restored.messages[0].content == "First msg"
        assert restored.messages[1].content == "Reply"

    def test_append_with_metadata(self, store):
        _init_store(store)
        session = Session(id="meta_001", depth=1, parent_id="root_001")

        store.append_message(
            "task_meta", session, "tool", "result data", tool_call_id="call_123"
        )

        restored = store.read_child_session("task_meta")
        assert restored is not None
        assert restored.messages[0].metadata["tool_call_id"] == "call_123"


class TestGetChildFilePath:
    def test_returns_absolute_path(self, store, tmp_path):
        _init_store(store)

        path = store.get_child_file_path("task_abc")

        assert path == str((tmp_path / "root_001" / "task_abc.json").resolve())


class TestSaveMainSession:
    def test_creates_main_json(self, store, root_session):
        _init_store(store)
        store.save_main_session(root_session)

        file_path = store.sessions_dir / "main.json"
        assert file_path.exists()

        data = json.loads(file_path.read_text())
        assert data["id"] == "root_001"
        assert data["depth"] == 0
        assert len(data["messages"]) == 2
