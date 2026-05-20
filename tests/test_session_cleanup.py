"""Tests for session cleanup (pruning) and per-session search cache."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from engine.runtime.agent_models import Session, Message
from engine.session_store import SessionStore
from engine.tools.custom.web_search import WebSearchTool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    """Create a SessionStore with a temporary root directory."""
    return SessionStore(str(tmp_path))


def _mock_config(tmp_path):
    """Return a mock config whose get_workspace_path() returns tmp_path."""
    config = MagicMock()
    config.get_workspace_path.return_value = tmp_path
    return config


def _create_session_dir(root: Path, session_id: str, mtime: float | None = None):
    """Helper: create a session directory with main.jsonl inside root."""
    session_dir = root / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "main.jsonl").write_text(
        json.dumps({"id": session_id, "depth": 0, "parent_id": None}) + "\n",
        encoding="utf-8",
    )
    if mtime is not None:
        os.utime(str(session_dir / "main.jsonl"), (mtime, mtime))
    return session_dir


# ===========================================================================
# TestSessionPruning
# ===========================================================================

class TestSessionPruning:
    """Tests for SessionStore.cleanup_old_sessions()."""

    def test_prune_after_save_keeps_max_sessions(self, store, tmp_path):
        """Creating 4 sessions then cleaning up with max=3 should leave 3."""
        base_time = 1700000000.0
        for i in range(4):
            _create_session_dir(tmp_path, f"chat_{i:03d}", mtime=base_time + i * 100)

        deleted = store.cleanup_old_sessions(max_sessions=3)

        assert deleted == 1
        remaining = store.list_sessions()
        assert len(remaining) == 3

    def test_no_prune_when_under_limit(self, store, tmp_path):
        """When session count <= max_sessions, nothing is deleted."""
        base_time = 1700000000.0
        _create_session_dir(tmp_path, "chat_000", mtime=base_time)
        _create_session_dir(tmp_path, "chat_001", mtime=base_time + 100)

        deleted = store.cleanup_old_sessions(max_sessions=3)

        assert deleted == 0
        assert len(store.list_sessions()) == 2

    def test_prune_removes_oldest_by_mtime(self, store, tmp_path):
        """Cleanup should remove the session with the oldest mtime."""
        base_time = 1700000000.0
        # Create 3 sessions with known mtime ordering
        _create_session_dir(tmp_path, "chat_oldest", mtime=base_time)
        _create_session_dir(tmp_path, "chat_mid", mtime=base_time + 200)
        _create_session_dir(tmp_path, "chat_newest", mtime=base_time + 400)
        # Create a 4th to trigger pruning
        _create_session_dir(tmp_path, "chat_extra", mtime=base_time + 600)

        deleted = store.cleanup_old_sessions(max_sessions=3)

        assert deleted == 1
        remaining = store.list_sessions()
        assert "chat_oldest" not in remaining
        assert len(remaining) == 3


# ===========================================================================
# TestPerSessionCache
# ===========================================================================

class TestPerSessionCache:
    """Tests for WebSearchTool._save_maintext() per-session cache routing."""

    @patch("engine.tools.custom.web_search.get_config")
    def test_cache_inside_session_dir(self, mock_get_config, tmp_path):
        """When context has a session, cache file goes under sessions/{id}/search_cache/."""
        mock_get_config.return_value = _mock_config(tmp_path)

        tool = WebSearchTool()
        session = Session(id="chat_test123", depth=0)
        context = {"session": session}

        result = tool._save_maintext("Some content here", "test query", 0, context=context)

        assert result is not None
        result_path = Path(result)
        assert result_path.exists()
        # Verify the file lives under the per-session cache dir
        expected_parent = tmp_path / "sessions" / "chat_test123" / "search_cache"
        assert result_path.parent == expected_parent

    @patch("engine.tools.custom.web_search.get_config")
    def test_returned_path_is_absolute(self, mock_get_config, tmp_path):
        """Returned file path should be absolute."""
        mock_get_config.return_value = _mock_config(tmp_path)

        tool = WebSearchTool()
        session = Session(id="chat_abs", depth=0)
        context = {"session": session}

        result = tool._save_maintext("Content", "abs query", 0, context=context)

        assert result is not None
        assert os.path.isabs(result)

    def test_cache_deleted_with_session(self, store, tmp_path):
        """Deleting a session should also remove its search_cache contents."""
        session = Session(id="chat_delcache", depth=0, messages=[
            Message(role="user", content="hi"),
        ])
        store.save(session)

        # Manually create a search_cache dir inside the session dir
        cache_dir = tmp_path / "chat_delcache" / "search_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "sample.md").write_text("cached content", encoding="utf-8")

        assert cache_dir.exists()

        store.delete("chat_delcache")

        assert not (tmp_path / "chat_delcache").exists()

    @patch("engine.tools.custom.web_search.get_config")
    def test_fallback_when_no_context(self, mock_get_config, tmp_path):
        """When context is None, file should be saved at workspace-level fallback path."""
        mock_get_config.return_value = _mock_config(tmp_path)

        tool = WebSearchTool()

        result = tool._save_maintext("Fallback content", "no ctx query", 0, context=None)

        assert result is not None
        result_path = Path(result)
        assert result_path.exists()
        # Should be under workspace/search_cache/, not under sessions/
        expected_parent = tmp_path / "search_cache"
        assert result_path.parent == expected_parent

    @patch("engine.tools.custom.web_search.get_config")
    def test_fallback_when_no_session_key(self, mock_get_config, tmp_path):
        """When context dict has no 'session' key, fallback to workspace-level path."""
        mock_get_config.return_value = _mock_config(tmp_path)

        tool = WebSearchTool()

        result = tool._save_maintext("No session key content", "empty ctx", 0, context={})

        assert result is not None
        result_path = Path(result)
        assert result_path.exists()
        expected_parent = tmp_path / "search_cache"
        assert result_path.parent == expected_parent


# ===========================================================================
# TestEdgeCases
# ===========================================================================

class TestEdgeCases:
    """Edge-case tests for cleanup and cache interactions."""

    def test_empty_session_list_cleanup(self, store, tmp_path):
        """Calling cleanup with zero sessions should not error."""
        deleted = store.cleanup_old_sessions(max_sessions=3)

        assert deleted == 0

    def test_concurrent_save_and_cleanup_sequence(self, store, tmp_path):
        """Simulate the chat.py flow: create session, save, then cleanup."""
        base_time = 1700000000.0
        # Pre-create 3 existing sessions
        for i in range(3):
            _create_session_dir(tmp_path, f"chat_old_{i:03d}", mtime=base_time + i * 100)

        # Create a new session via save (like chat.py does)
        new_session = Session(id="chat_new_fresh", depth=0, messages=[
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
        ])
        store.save(new_session)

        # Give the new session a newer mtime so it survives pruning
        new_jsonl = tmp_path / "chat_new_fresh" / "main.jsonl"
        os.utime(str(new_jsonl), (base_time + 1000, base_time + 1000))

        # Cleanup with max=3 should remove the oldest pre-existing session
        deleted = store.cleanup_old_sessions(max_sessions=3)

        assert deleted == 1
        remaining = store.list_sessions()
        assert len(remaining) == 3
        assert "chat_new_fresh" in remaining
