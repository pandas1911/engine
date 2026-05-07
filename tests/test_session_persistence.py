"""Tests for SessionStore JSONL file persistence layer."""

import json
from pathlib import Path

import pytest

from engine.runtime.agent_models import Session, Message
from engine.session_store import SessionStore, ChildSessionInfo


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


def _write_child(store, task_id, session):
    """Helper: create a child JSONL file from a session."""
    store.create_file(task_id, session)
    for msg in session.messages:
        store.append_line(task_id, msg)


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


class TestCreateFile:
    def test_creates_task_jsonl(self, store, child_session):
        _init_store(store)
        store.create_file("task_abc123", child_session)

        file_path = store.sessions_dir / "task_abc123.jsonl"
        assert file_path.exists()

    def test_header_contains_session_metadata(self, store, child_session):
        _init_store(store)
        store.create_file("task_abc123", child_session)

        raw = (store.sessions_dir / "task_abc123.jsonl").read_text(encoding="utf-8")
        lines = raw.strip().splitlines()
        header = json.loads(lines[0])

        assert header["id"] == "child_001"
        assert header["depth"] == 1
        assert header["parent_id"] == "root_001"

    def test_header_only_no_messages(self, store, child_session):
        _init_store(store)
        store.create_file("task_abc123", child_session)

        raw = (store.sessions_dir / "task_abc123.jsonl").read_text(encoding="utf-8")
        lines = [l for l in raw.splitlines() if l.strip()]
        assert len(lines) == 1  # header only


class TestAppendLine:
    def test_appends_message_as_json_line(self, store):
        _init_store(store)
        session = Session(id="append_001", depth=1, parent_id="root_001")
        store.create_file("task_append", session)

        msg = Message(role="user", content="First msg")
        store.append_line("task_append", msg)

        raw = (store.sessions_dir / "task_append.jsonl").read_text(encoding="utf-8")
        lines = [l for l in raw.splitlines() if l.strip()]
        assert len(lines) == 2  # header + 1 message

        msg_data = json.loads(lines[1])
        assert msg_data["role"] == "user"
        assert msg_data["content"] == "First msg"

    def test_multiple_appends(self, store):
        _init_store(store)
        session = Session(id="multi_001", depth=1, parent_id="root_001")
        store.create_file("task_multi", session)

        store.append_line("task_multi", Message(role="user", content="Q1"))
        store.append_line("task_multi", Message(role="assistant", content="A1"))
        store.append_line("task_multi", Message(role="user", content="Q2"))

        raw = (store.sessions_dir / "task_multi.jsonl").read_text(encoding="utf-8")
        lines = [l for l in raw.splitlines() if l.strip()]
        assert len(lines) == 4  # header + 3 messages


class TestRewriteFile:
    def test_full_rewrite_preserves_all_data(self, store, child_session):
        _init_store(store)
        _write_child(store, "task_rewrite", child_session)

        # Add extra messages in memory
        child_session.add_message("user", "Extra question")
        child_session.add_message("assistant", "Extra answer")

        store.rewrite_file("task_rewrite", child_session)

        restored = store.read_child_session("task_rewrite")
        assert restored is not None
        assert len(restored.messages) == 4  # original 2 + 2 new


class TestRoundTrip:
    def test_serialize_deserialize_roundtrip(self, store, child_session):
        _init_store(store)
        _write_child(store, "task_xyz", child_session)

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
        _write_child(store, "task_alpha", child_session)
        _write_child(store, "task_beta", child_session)

        children = store.list_children()

        assert len(children) == 2
        task_ids = {c.task_id for c in children}
        assert task_ids == {"task_alpha", "task_beta"}

    def test_child_info_metadata(self, store, child_session):
        _init_store(store)
        _write_child(store, "task_abc", child_session)

        info = store.list_children()[0]

        assert isinstance(info, ChildSessionInfo)
        assert info.task_id == "task_abc"
        assert info.message_count == 2
        assert info.file_size_bytes > 0
        assert "task_abc.jsonl" in info.file_path

    def test_empty_directory(self, store):
        _init_store(store)

        assert store.list_children() == []

    def test_prefers_jsonl_over_json(self, store, child_session):
        _init_store(store)
        _write_child(store, "task_both", child_session)
        # Also create a legacy .json file
        json_path = store.sessions_dir / "task_both.json"
        json_path.write_text('{"id":"old","depth":0,"messages":[]}', encoding="utf-8")

        children = store.list_children()
        assert len(children) == 1
        assert children[0].message_count == 2  # from .jsonl, not .json


class TestPartialSession:
    def test_partial_session_valid_jsonl(self, store, child_session):
        _init_store(store)
        child_session.add_message("user", "Partial work")
        _write_child(store, "task_partial", child_session)

        restored = store.read_child_session("task_partial")

        assert restored is not None
        assert len(restored.messages) == 3

    def test_malformed_line_skipped(self, store):
        _init_store(store)
        # Write header + valid message + malformed line
        lines = [
            json.dumps({"id": "sess_mal", "depth": 1, "parent_id": "root_001"}),
            json.dumps({"role": "user", "content": "valid", "metadata": {}}),
            "not valid json{{{{",
        ]
        bad_file = store.sessions_dir / "task_malformed.jsonl"
        bad_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        restored = store.read_child_session("task_malformed")
        assert restored is not None
        assert len(restored.messages) == 1  # only the valid message

    def test_corrupted_header_returns_none(self, store):
        _init_store(store)
        bad_file = store.sessions_dir / "task_bad_header.jsonl"
        bad_file.write_text("not json at all\n", encoding="utf-8")

        assert store.read_child_session("task_bad_header") is None

    def test_corrupted_file_list_children(self, store):
        _init_store(store)
        bad_file = store.sessions_dir / "task_bad.jsonl"
        bad_file.write_text("not json at all", encoding="utf-8")

        children = store.list_children()
        assert len(children) == 1
        # JSONL list_children counts non-empty lines minus header (1 line)
        # A single corrupted line yields max(0, 1-1) = 0
        assert children[0].message_count == 0


class TestLegacyJsonCompat:
    def test_reads_old_json_format(self, store):
        _init_store(store)
        legacy_data = {
            "id": "sess_legacy",
            "depth": 1,
            "parent_id": "root_001",
            "messages": [
                {"role": "user", "content": "Legacy msg", "metadata": {}},
                {"role": "assistant", "content": "Legacy reply", "metadata": {}},
            ],
        }
        json_path = store.sessions_dir / "task_legacy.json"
        json_path.write_text(json.dumps(legacy_data), encoding="utf-8")

        restored = store.read_child_session("task_legacy")
        assert restored is not None
        assert restored.id == "sess_legacy"
        assert len(restored.messages) == 2
        assert restored.messages[0].content == "Legacy msg"


class TestAppFacingApi:
    def test_save_creates_main_jsonl(self, store, root_session):
        path = store.save(root_session)

        assert path.exists()
        assert path.name == "main.jsonl"

    def test_load_roundtrip(self, store, root_session):
        store.save(root_session)
        loaded = store.load(root_session.id)

        assert loaded is not None
        assert loaded.id == root_session.id
        assert len(loaded.messages) == 2

    def test_delete_removes_directory(self, store, root_session):
        store.save(root_session)
        assert store.delete(root_session.id) is True
        assert store.load(root_session.id) is None

    def test_list_sessions(self, store, root_session):
        store.save(root_session)
        sessions = store.list_sessions()

        assert root_session.id in sessions

    def test_load_nonexistent_returns_none(self, store):
        assert store.load("nonexistent") is None

    def test_delete_nonexistent_returns_false(self, store):
        assert store.delete("nonexistent") is False
