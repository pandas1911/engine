"""Tests for logging warning functionality and tool color change."""

from unittest.mock import MagicMock, patch

from engine.logging.sink import LogEntry, Logger, TerminalFormatter


def test_logger_warning_creates_warning_entry():
    """Logger.warning() should produce a LogEntry with level='warning'."""
    logger = Logger(terminal=False)
    captured = []
    logger._output = lambda entry: captured.append(entry)
    logger.warning(
        "test-agent",
        "test warning message",
        task_id="t1",
        state="RUNNING",
        depth=1,
        event_type="truncation",
    )
    assert len(captured) == 1
    entry = captured[0]
    assert entry.level == "warning"
    assert entry.agent_label == "test-agent"
    assert entry.message == "test warning message"
    assert entry.agent_id == "t1"
    assert entry.state == "RUNNING"
    assert entry.depth == 1
    assert entry.event_type == "truncation"


def test_terminal_formatter_warning_yellow():
    """Warning log entries should render in YELLOW."""
    formatter = TerminalFormatter()
    entry = LogEntry(
        timestamp="2025-01-01T00:00:00.000",
        level="warning",
        agent_id="x",
        agent_label="agent",
        depth=0,
        state="RUNNING",
        event_type="warning",
        message="something warned",
    )
    output = formatter.format(entry)
    assert "\033[33m" in output


def test_terminal_formatter_tool_magenta():
    """Tool log entries should render in MAGENTA (not YELLOW)."""
    formatter = TerminalFormatter()
    entry = LogEntry(
        timestamp="2025-01-01T00:00:00.000",
        level="info",
        agent_id="x",
        agent_label="agent",
        depth=0,
        state="RUNNING",
        event_type="tool",
        message="ran a tool",
    )
    output = formatter.format(entry)
    assert "\033[35m" in output
    assert "\033[33m" not in output


def test_terminal_formatter_error_red_regression():
    """Error log entries should still render in RED (regression guard)."""
    formatter = TerminalFormatter()
    entry = LogEntry(
        timestamp="2025-01-01T00:00:00.000",
        level="error",
        agent_id="x",
        agent_label="agent",
        depth=0,
        state="ERROR",
        event_type="error",
        message="something broke",
    )
    output = formatter.format(entry)
    assert "\033[31m" in output
