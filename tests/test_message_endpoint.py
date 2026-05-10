"""Tests for POST /api/chat/message endpoint."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app._state import set_active_session, clear_active_session


client = TestClient(app)


class TestMessageEndpoint:
    def setup_method(self):
        clear_active_session()

    def teardown_method(self):
        clear_active_session()

    def test_message_no_active_session_404(self):
        response = client.post("/api/chat/message", json={"message": "test"})
        assert response.status_code == 404
        assert "error" in response.json()

    def test_message_agent_completed_409(self):
        mock_mgr = MagicMock()
        mock_mgr.interject.return_value = "rejected"
        set_active_session(
            session_id="test_session",
            session_manager=mock_mgr,
            event_queue=[],
            done_event=MagicMock(),
        )
        response = client.post("/api/chat/message", json={"message": "test"})
        assert response.status_code == 409
        assert "error" in response.json()

    def test_message_accepted_200(self):
        mock_mgr = MagicMock()
        mock_mgr.interject.return_value = "accepted"
        set_active_session(
            session_id="test_session",
            session_manager=mock_mgr,
            event_queue=[],
            done_event=MagicMock(),
        )
        response = client.post("/api/chat/message", json={"message": "Status?"})
        assert response.status_code == 200
        assert response.json()["status"] == "accepted"

    def test_message_queued_200(self):
        mock_mgr = MagicMock()
        mock_mgr.interject.return_value = "queued"
        set_active_session(
            session_id="test_session",
            session_manager=mock_mgr,
            event_queue=[],
            done_event=MagicMock(),
        )
        response = client.post("/api/chat/message", json={"message": "hello"})
        assert response.status_code == 200
        assert response.json()["status"] == "queued"
