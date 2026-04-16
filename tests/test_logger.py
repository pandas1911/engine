import asyncio
import json
import os
import pytest

BLUE = "\033[34m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
RESET = "\033[0m"


@pytest.fixture(autouse=True)
def reset_logger():
    import engine.logger as logger_mod
    logger_mod._logger = None
    yield
    logger_mod._logger = None


def test_logger_module_import():
    from engine.logger import get_logger, init_logger, stop_logger, Logger, LoggerInterface, LogEntry
    assert callable(get_logger)
    assert callable(init_logger)
    assert callable(stop_logger)


def test_terminal_output_has_ansi_colors(capsys):
    from engine.logger import get_logger
    logger = get_logger()
    logger.info("Root", "test", task_id="t1", state="running", depth=0)
    captured = capsys.readouterr()
    assert BLUE in captured.out
    assert CYAN in captured.out
    assert RESET in captured.out

    logger.info("Sub", "test", task_id="t2", state="running", depth=1)
    captured = capsys.readouterr()
    assert GREEN in captured.out

    logger.tool("Root", "tool test", task_id="t1", state="running", depth=0, tool_name="spawn")
    captured = capsys.readouterr()
    assert YELLOW in captured.out
    assert "Root(spawn)" in captured.out

    logger.error("Root", "error", task_id="t1", state="error", depth=0)
    captured = capsys.readouterr()
    assert RED in captured.out


@pytest.mark.asyncio
async def test_jsonl_file_created(tmp_path):
    from engine.logger import init_logger, get_logger, stop_logger
    init_logger(log_dir=str(tmp_path))
    logger = get_logger()
    logger.info("Root", "test", task_id="t1", state="running", depth=0)
    await stop_logger()

    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1


@pytest.mark.asyncio
async def test_jsonl_all_required_fields(tmp_path):
    from engine.logger import init_logger, get_logger, stop_logger
    init_logger(log_dir=str(tmp_path))
    logger = get_logger()
    logger.info("Root", "test message", task_id="t1", state="running", depth=0, event_type="test_event", data={"key": "value"})
    await stop_logger()

    files = list(tmp_path.glob("*.jsonl"))
    with open(files[0]) as f:
        entry = json.loads(f.readline())

    required_fields = ["timestamp", "level", "agent_id", "agent_label", "depth", "state", "event_type", "message"]
    for field in required_fields:
        assert field in entry, "Missing field: {}".format(field)
    assert entry["level"] == "info"
    assert entry["agent_label"] == "Root"
    assert entry["message"] == "test message"


@pytest.mark.asyncio
async def test_async_flush_all_messages(tmp_path):
    from engine.logger import init_logger, get_logger, stop_logger
    init_logger(log_dir=str(tmp_path))
    logger = get_logger()
    for i in range(100):
        logger.info("Root", "msg {}".format(i), task_id="t1", state="running", depth=0)
    await stop_logger()

    files = list(tmp_path.glob("*.jsonl"))
    with open(files[0]) as f:
        lines = f.readlines()
    assert len(lines) == 100, "Expected 100 lines, got {}".format(len(lines))


@pytest.mark.asyncio
async def test_custom_log_directory(tmp_path):
    from engine.logger import init_logger, get_logger, stop_logger
    custom_dir = str(tmp_path / "custom_logs")
    init_logger(log_dir=custom_dir)
    logger = get_logger()
    logger.info("Root", "test", task_id="t1", state="running", depth=0)
    await stop_logger()

    assert os.path.isdir(custom_dir)
    files = [f for f in os.listdir(custom_dir) if f.endswith(".jsonl")]
    assert len(files) >= 1


def test_sync_buffer_before_loop(tmp_path):
    from engine.logger import get_logger, init_logger, stop_logger

    async def _run():
        logger = get_logger()
        logger.info("Root", "buffered msg", task_id="t1", state="running", depth=0)
        init_logger(log_dir=str(tmp_path))
        logger.info("Root", "after init msg", task_id="t1", state="running", depth=0)
        await stop_logger()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run())
    finally:
        loop.close()

    files = list(tmp_path.glob("*.jsonl"))
    with open(files[0]) as f:
        lines = f.readlines()
    assert len(lines) == 2, "Expected 2 lines (1 buffered + 1 after init), got {}".format(len(lines))


def test_state_change_logging(capsys):
    from engine.logger import get_logger
    logger = get_logger()
    logger.state_change("Root", "idle", "running", "start", task_id="t1", depth=0)
    captured = capsys.readouterr()
    assert BLUE in captured.out
    assert "state_change" in captured.out


def test_tool_logging(capsys):
    from engine.logger import get_logger
    logger = get_logger()
    logger.tool("Root", "Executing spawn", task_id="t1", state="running", depth=0, tool_name="spawn")
    captured = capsys.readouterr()
    assert YELLOW in captured.out
    assert "Root(spawn)" in captured.out


def test_error_logging(capsys):
    from engine.logger import get_logger
    logger = get_logger()
    logger.error("Root", "Something failed", task_id="t1", state="error", depth=0, event_type="tool_error")
    captured = capsys.readouterr()
    assert RED in captured.out
    assert "Something failed" in captured.out


@pytest.mark.asyncio
async def test_graceful_shutdown_without_init():
    from engine.logger import stop_logger
    import engine.logger as logger_mod
    logger_mod._logger = None
    await stop_logger()


@pytest.mark.asyncio
async def test_init_logger_with_config(tmp_path):
    from engine.logger import init_logger, get_logger, stop_logger
    from engine.config import Config
    config = Config(api_key="test", base_url="http://test", model="test", log_dir=str(tmp_path / "config_logs"))
    init_logger(config=config)
    logger = get_logger()
    logger.info("Root", "config test", task_id="t1", state="running", depth=0)
    await stop_logger()

    assert os.path.isdir(str(tmp_path / "config_logs"))
    files = [f for f in os.listdir(str(tmp_path / "config_logs")) if f.endswith(".jsonl")]
    assert len(files) >= 1
