"""Tests for Engine singleton and reset."""

from unittest.mock import MagicMock, patch

from engine.runner import Engine


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
