"""Logging package for the agent system.

Provides:
- Core logging infrastructure (Logger, formatters, file handler)
- AgentLogHelper for agent-context-aware logging
- Public API: get_logger, init_logger, stop_logger, AgentLogHelper
"""

from engine.logging.sink import (
    Logger,
    LoggerInterface,
    LogEntry,
    TerminalFormatter,
    JSONFormatter,
    AsyncFileHandler,
    get_logger,
    init_logger,
    stop_logger,
)
from engine.logging.agent_log import AgentLogHelper

__all__ = [
    "Logger",
    "LoggerInterface",
    "LogEntry",
    "TerminalFormatter",
    "JSONFormatter",
    "AsyncFileHandler",
    "get_logger",
    "init_logger",
    "stop_logger",
    "AgentLogHelper",
]
