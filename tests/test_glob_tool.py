"""Tests for GlobTool — file pattern matching tool."""

import asyncio

import pytest

from engine.tools.builtin.glob_tool import GlobTool


async def _execute(tool, args):
    return await tool.execute(args, {})


@pytest.fixture
def glob_dir(tmp_path):
    """Create a temporary directory with files for glob testing."""
    (tmp_path / "app.py").write_text("code")
    (tmp_path / "utils.py").write_text("code")
    (tmp_path / "readme.md").write_text("text")
    (tmp_path / "data.txt").write_text("text")
    sub = tmp_path / "src"
    sub.mkdir()
    (sub / "main.py").write_text("code")
    (sub / "helper.py").write_text("code")
    (sub / "config.json").write_text("{}")
    return tmp_path


class TestGlobToolBasic:
    def test_glob_py_files(self, glob_dir):
        result = asyncio.run(_execute(GlobTool(), {"pattern": "**/*.py", "path": str(glob_dir)}))
        assert "<pattern>**/*.py</pattern>" in result
        assert "<files>" in result
        assert "<summary>" in result
        assert "app.py" in result
        assert "utils.py" in result
        assert "src/main.py" in result or "main.py" in result

    def test_glob_specific_dir(self, glob_dir):
        result = asyncio.run(_execute(GlobTool(), {"pattern": "*.py", "path": str(glob_dir / "src")}))
        assert "helper.py" in result
        assert "app.py" not in result  # Not in src dir

    def test_glob_no_matches(self, glob_dir):
        result = asyncio.run(_execute(GlobTool(), {"pattern": "*.xyz", "path": str(glob_dir)}))
        assert "Found 0 files" in result or "0 file" in result

    def test_glob_missing_pattern(self):
        result = asyncio.run(_execute(GlobTool(), {"pattern": ""}))
        assert "Error" in result

    def test_glob_output_format(self, glob_dir):
        result = asyncio.run(_execute(GlobTool(), {"pattern": "*.md", "path": str(glob_dir)}))
        assert "<pattern>*.md</pattern>" in result
        assert f"<path>{glob_dir}</path>" in result
        assert "<files>" in result
        assert "readme.md" in result
        assert "<summary>" in result

    def test_glob_default_path(self, glob_dir, monkeypatch):
        """Test that default path is cwd."""
        monkeypatch.chdir(glob_dir)
        result = asyncio.run(_execute(GlobTool(), {"pattern": "*.md"}))
        assert "readme.md" in result
