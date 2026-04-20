from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import asyncio
import json
import sys

from engine.config import Config


class LoggerInterface(ABC):
    @abstractmethod
    def info(self, agent_label: str, message: str, **kwargs) -> None:
        pass

    @abstractmethod
    def error(self, agent_label: str, message: str, **kwargs) -> None:
        pass

    @abstractmethod
    def warning(self, agent_label: str, message: str, **kwargs) -> None:
        pass

    @abstractmethod
    def tool(self, agent_label: str, message: str, **kwargs) -> None:
        pass

    @abstractmethod
    def state_change(
        self, agent_label: str, from_state: str, to_state: str, trigger: str, **kwargs
    ) -> None:
        pass


@dataclass
class LogEntry:
    timestamp: str
    level: str
    agent_id: str
    agent_label: str
    depth: int
    state: str
    event_type: str
    message: str
    data: Optional[Dict[str, Any]] = None
    tool_name: Optional[str] = None


class TerminalFormatter:
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    RESET = "\033[0m"

    def format(self, entry: LogEntry) -> str:
        if entry.level == "error":
            color = self.RED
        elif entry.level == "warning":
            color = self.YELLOW
        elif entry.event_type == "tool":
            color = self.MAGENTA
        elif entry.depth > 0:
            color = self.GREEN
        else:
            color = self.BLUE

        dt = entry.timestamp
        if len(dt) >= 23:
            time_part = dt[11:23]
        else:
            time_part = dt

        label = entry.agent_label
        if entry.tool_name:
            label = "{}({})".format(entry.agent_label, entry.tool_name)

        if entry.level == "error" or entry.event_type == "state_change":
            prefix = "{}[{}|{} {}]{}".format(
                color, label, entry.agent_id, time_part, self.RESET
            )
        else:
            prefix = "{}[{}|{} {}]{} {}[{}]{}".format(
                color,
                label,
                entry.agent_id,
                time_part,
                self.RESET,
                self.CYAN,
                entry.state.upper(),
                self.RESET,
            )

        parts = [prefix, entry.message]
        if entry.data:
            parts.append(json.dumps(entry.data, ensure_ascii=False))
        return " ".join(parts)


class JSONFormatter:
    def format(self, entry: LogEntry) -> str:
        return json.dumps(asdict(entry), ensure_ascii=False)


class AsyncFileHandler:
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = self.log_dir / "run_{}.jsonl".format(timestamp)
        self.queue = None  # type: Optional[asyncio.Queue]
        self._writer_task = None  # type: Optional[asyncio.Task]
        self._loop = None  # type: Optional[asyncio.AbstractEventLoop]

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self.queue = asyncio.Queue()
        self._writer_task = loop.create_task(self._writer())

    async def _writer(self) -> None:
        with open(self.file_path, "a", encoding="utf-8") as f:
            while True:
                assert self.queue is not None
                line = await self.queue.get()
                if line is None:
                    self.queue.task_done()
                    break
                await self._write_line(f, line)
                self.queue.task_done()

    async def _write_line(self, f, line: str) -> None:
        assert self._loop is not None

        def _sync_write() -> None:
            f.write(line + "\n")
            f.flush()

        await self._loop.run_in_executor(None, _sync_write)

    def _on_task_done(self, task: asyncio.Task) -> None:
        try:
            task.result()
        except Exception as exc:
            sys.stderr.write("AsyncFileHandler writer error: {}\n".format(exc))

    def emit(self, line: str) -> None:
        if self.queue is not None:
            self.queue.put_nowait(line)

    async def stop(self) -> None:
        if self.queue is None:
            return
        self.queue.put_nowait(None)
        if self._writer_task is not None:
            self._writer_task.add_done_callback(self._on_task_done)
            await self.queue.join()
            if not self._writer_task.done():
                await self._writer_task


class Logger(LoggerInterface):
    def __init__(self, terminal: bool = True, file_handler: Optional[AsyncFileHandler] = None):
        self.terminal = terminal
        self.file_handler = file_handler
        self.terminal_formatter = TerminalFormatter()
        self.json_formatter = JSONFormatter()
        self._sync_buffer = deque()  # type: deque
        self._initialized = False

    def _make_entry(
        self,
        level: str,
        agent_label: str,
        message: str,
        event_type: str,
        task_id: str,
        state: str,
        depth: int,
        data: Optional[Dict[str, Any]] = None,
        tool_name: Optional[str] = None,
    ) -> LogEntry:
        return LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            agent_id=task_id,
            agent_label=agent_label,
            depth=depth,
            state=state,
            event_type=event_type,
            message=message,
            data=data,
            tool_name=tool_name,
        )

    def _output(self, entry: LogEntry) -> None:
        if self.terminal:
            print(self.terminal_formatter.format(entry))

        json_line = self.json_formatter.format(entry)
        if self._initialized and self.file_handler is not None:
            self.file_handler.emit(json_line)
        else:
            self._sync_buffer.append(json_line)

    def info(self, agent_label: str, message: str, **kwargs) -> None:
        task_id = kwargs.get("task_id", "unknown")
        state = kwargs.get("state", "idle")
        depth = kwargs.get("depth", 0)
        event_type = kwargs.get("event_type", "info")
        data = kwargs.get("data")
        entry = self._make_entry(
            level="info",
            agent_label=agent_label,
            message=message,
            event_type=event_type,
            task_id=task_id,
            state=state,
            depth=depth,
            data=data,
        )
        self._output(entry)

    def error(self, agent_label: str, message: str, **kwargs) -> None:
        task_id = kwargs.get("task_id", "unknown")
        state = kwargs.get("state", "error")
        depth = kwargs.get("depth", 0)
        event_type = kwargs.get("event_type", "error")
        data = kwargs.get("data")
        entry = self._make_entry(
            level="error",
            agent_label=agent_label,
            message=message,
            event_type=event_type,
            task_id=task_id,
            state=state,
            depth=depth,
            data=data,
        )
        self._output(entry)

    def warning(self, agent_label: str, message: str, **kwargs) -> None:
        task_id = kwargs.get("task_id", "unknown")
        state = kwargs.get("state", "idle")
        depth = kwargs.get("depth", 0)
        event_type = kwargs.get("event_type", "warning")
        data = kwargs.get("data")
        entry = self._make_entry(
            level="warning",
            agent_label=agent_label,
            message=message,
            event_type=event_type,
            task_id=task_id,
            state=state,
            depth=depth,
            data=data,
        )
        self._output(entry)

    def tool(self, agent_label: str, message: str, **kwargs) -> None:
        task_id = kwargs.get("task_id", "unknown")
        state = kwargs.get("state", "running")
        depth = kwargs.get("depth", 0)
        tool_name = kwargs.get("tool_name", "unknown")
        data = kwargs.get("data", {})
        if tool_name and "tool_name" not in data:
            data = dict(data)
            data["tool_name"] = tool_name
        entry = self._make_entry(
            level="info",
            agent_label=agent_label,
            message=message,
            event_type="tool",
            task_id=task_id,
            state=state,
            depth=depth,
            data=data,
            tool_name=tool_name,
        )
        self._output(entry)

    def state_change(
        self, agent_label: str, from_state: str, to_state: str, trigger: str, **kwargs
    ) -> None:
        task_id = kwargs.get("task_id", "unknown")
        state = kwargs.get("state", to_state)
        depth = kwargs.get("depth", 0)
        data = {
            "from_state": from_state,
            "to_state": to_state,
            "trigger": trigger,
        }
        entry = self._make_entry(
            level="info",
            agent_label=agent_label,
            message="state_change",
            event_type="state_change",
            task_id=task_id,
            state=state,
            depth=depth,
            data=data,
        )
        self._output(entry)

    def init_async(self, loop: asyncio.AbstractEventLoop) -> None:
        if self.file_handler is not None:
            self.file_handler.start(loop)
        self._initialized = True
        while self._sync_buffer:
            line = self._sync_buffer.popleft()
            if self.file_handler is not None:
                self.file_handler.emit(line)

    async def stop(self) -> None:
        if self.file_handler is not None:
            await self.file_handler.stop()


_logger = None  # type: Optional[Logger]


def get_logger() -> Logger:
    global _logger
    if _logger is None:
        _logger = Logger(terminal=True, file_handler=None)
    return _logger


async def stop_logger() -> None:
    global _logger
    if _logger is not None:
        await _logger.stop()


def init_logger(
    log_dir: Optional[str] = None, config: Optional[Config] = None
) -> None:
    global _logger

    target_dir = log_dir
    if target_dir is None and config is not None:
        target_dir = getattr(config, "log_dir", None)
    if target_dir is None:
        target_dir = "logs"

    file_handler = AsyncFileHandler(target_dir)
    if _logger is None:
        _logger = Logger(terminal=True, file_handler=file_handler)
    else:
        _logger.file_handler = file_handler

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()

    _logger.init_async(loop)
