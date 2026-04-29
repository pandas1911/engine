"""Unit tests for tool system refactoring: ToolPack, SpawnTool, Config.tools, AgentTask, context dict."""

import asyncio
import json
import os
import tempfile

import pytest

from engine.config import Config, ConfigLoader
from engine.tools.base import FunctionTool, ToolRegistry, ToolRegistrationError
from engine.tools.pack import ToolPack
from engine.subagent.spawn import SpawnTool
from engine.subagent.subagent_models import AgentTask


# ---- Helpers ----


def _make_tool(name="test_tool", description="A test tool"):
    return FunctionTool(name=name, description=description, parameters={}, fn=lambda _args, _ctx: "ok")


# ---- ToolPack tests ----


class TestToolPack:
    def test_construction_and_basic_ops(self):
        t1 = _make_tool("tool_a")
        t2 = _make_tool("tool_b")
        pack = ToolPack([t1, t2])
        assert len(pack) == 2
        assert "tool_a" in pack
        assert pack.get("tool_a") is t1
        assert pack.get("nonexistent") is None

    def test_rejects_duplicates(self):
        t1 = _make_tool("dup")
        t2 = _make_tool("dup")
        with pytest.raises(ToolRegistrationError, match="already registered"):
            ToolPack([t1, t2])

    def test_rejects_empty_name(self):
        with pytest.raises(ToolRegistrationError, match="cannot be empty"):
            _make_tool("")

    def test_immutability(self):
        """ToolPack must not expose mutable registry methods directly."""
        pack = ToolPack([_make_tool("a")])
        # ToolPack has no register/unregister/all_tools/tool_names as public methods
        assert not hasattr(pack, "register")
        assert not hasattr(pack, "unregister")
        assert not hasattr(pack, "all_tools")
        assert not hasattr(pack, "tool_names")

    def test_wraps_tool_registry_internally(self):
        pack = ToolPack([_make_tool("x")])
        assert isinstance(pack._registry, ToolRegistry)

    def test_depth_gating_schema_filter(self):
        """Spawn schema filtered out when session depth >= max_depth."""
        from engine.runtime.agent_models import Session

        import engine.tools.pack as pack_mod

        spawn = _make_tool("spawn")
        other = _make_tool("search")
        pack = ToolPack([spawn, other])

        cfg = Config(primary="p/m", max_depth=3)
        original = pack_mod.get_config
        pack_mod.get_config = lambda: cfg
        try:
            # depth < max_depth -> spawn visible
            schemas_ok = pack.get_schemas(Session(id="s1", depth=1))
            names_ok = [s["function"]["name"] for s in schemas_ok]
            assert "spawn" in names_ok

            # depth >= max_depth -> spawn hidden
            schemas_max = pack.get_schemas(Session(id="s2", depth=3))
            names_max = [s["function"]["name"] for s in schemas_max]
            assert "spawn" not in names_max
            assert "search" in names_max
        finally:
            pack_mod.get_config = original

    def test_release_spawn_noop_when_no_spawn(self):
        """release_spawn should not raise even when no spawn tool is present."""
        pack = ToolPack([_make_tool("a")])
        pack.release_spawn("task_123")


# ---- Config.tools tests ----


class TestConfigTools:
    def test_tools_field_default(self):
        c = Config(primary="p/m")
        assert c.tools == {}
        assert c.is_tool_enabled("anything") is True

    def test_tools_field_enabled_disabled(self):
        c = Config(primary="p/m", tools={"web_search": False, "web_fetch": True})
        assert c.is_tool_enabled("web_search") is False
        assert c.is_tool_enabled("web_fetch") is True
        assert c.is_tool_enabled("spawn") is True

    def test_config_validation_rejects_non_dict(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(
            {
                "providers": {
                    "p": {"api_key": "k", "base_url": "u", "models": {"m": {}}}
                },
                "primary": "p/m",
                "tools": "invalid",
            },
            f,
        )
        f.close()
        try:
            with pytest.raises(ValueError, match="tools"):
                ConfigLoader.load_from_json(f.name)
        finally:
            os.unlink(f.name)


# ---- ToolRegistry clean tests ----


class TestToolRegistryClean:
    def test_no_spawn_special_casing(self):
        """ToolRegistry treats 'spawn' like any other tool."""
        r = ToolRegistry()
        t = _make_tool("spawn")
        r.register(t)
        assert r.get("spawn") is t
        assert len(r) == 1
        r.unregister("spawn")
        assert len(r) == 0

    def test_register_spawn_and_clone_removed(self):
        """ToolRegistry must not have spawn-specific or clone methods."""
        r = ToolRegistry()
        assert not hasattr(r, "register_spawn")
        assert not hasattr(r, "clone")


# ---- SpawnTool tests ----


class TestSpawnTool:
    def test_lazy_cache_init(self):
        tool = SpawnTool()
        assert len(tool._managers) == 0
        assert hasattr(tool, "release")

    def test_release_cleans_up(self):
        tool = SpawnTool()
        tool._managers["task_123"] = "fake"
        tool.release("task_123")
        assert "task_123" not in tool._managers

    def test_depth_gating_rejects(self):
        """SpawnTool rejects execution when session.depth >= max_depth."""
        from engine.runtime.agent_models import Session

        import engine.config as cfg_mod

        class MockAgent:
            task_id = "task_001"
            label = "Root"

        tool = SpawnTool()
        cfg = Config(primary="p/m", max_depth=3)
        original = cfg_mod.get_config
        cfg_mod.get_config = lambda: cfg
        try:
            session = Session(id="s", depth=3)
            ctx = {"session": session, "parent_agent": MockAgent()}
            result = asyncio.run(tool.execute({"task": "test"}, ctx))
            assert "Spawn Failed" in result or "ERROR" in result
        finally:
            cfg_mod.get_config = original


# ---- AgentTask rename test ----


class TestAgentTaskRename:
    def test_import_works(self):
        task = AgentTask(
            task_id="t1", session_id="s1", task_description="test", parent_agent=None
        )
        assert task.task_id == "t1"
        assert task.result is None

    def test_no_subagent_task_import(self):
        import engine.subagent.subagent_models as models

        assert not hasattr(models, "SubagentTask")


# ---- Context dict test ----


class TestContextDict:
    def test_context_has_two_keys(self):
        """Verify context dict in agent._execute_tool has exactly 2 keys."""
        import inspect

        from engine.runtime.agent import Agent

        source = inspect.getsource(Agent._execute_tool)
        # Extract the context dict literal
        context_section = source.split("context = {")[1].split("}")[0]
        assert '"session"' in context_section
        assert '"parent_agent"' in context_section
        assert '"config"' not in context_section
        assert '"agent_factory"' not in context_section
