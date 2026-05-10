"""Tests for GrepTool — content search using regex."""

import asyncio

from engine.tools.builtin.grep import GrepTool


async def _execute(tool: GrepTool, args: dict) -> str:
    """Helper to run execute synchronously in tests."""
    return await tool.execute(args, {})


def run(tool: GrepTool, args: dict) -> str:
    return asyncio.run(_execute(tool, args))


class TestGrepToolBasicSearch:
    """Test basic regex searching."""

    def test_finds_matches(self, tmp_path):
        (tmp_path / "app.py").write_text("def hello():\n    pass\n\ndef world():\n    pass\n")
        result = run(GrepTool(), {"pattern": "def hello", "path": str(tmp_path)})
        assert "<pattern>def hello</pattern>" in result
        assert "app.py" in result
        assert "<matches>" in result
        assert "<summary>" in result

    def test_match_format_is_file_line_content(self, tmp_path):
        (tmp_path / "app.py").write_text("def hello():\n    pass\n")
        result = run(GrepTool(), {"pattern": "def hello", "path": str(tmp_path)})
        # Match format: file:line: content
        assert ":1: " in result
        assert "def hello():" in result

    def test_multiple_matches_across_files(self, tmp_path):
        (tmp_path / "a.py").write_text("import os\n")
        (tmp_path / "b.py").write_text("import sys\n")
        result = run(GrepTool(), {"pattern": "import ", "path": str(tmp_path)})
        assert "<summary>" in result
        assert "2 matches" in result
        assert "2 file(s)" in result


class TestGrepToolOutputFormat:
    """Test XML output structure."""

    def test_output_has_all_xml_tags(self, tmp_path):
        (tmp_path / "x.py").write_text("hello\n")
        result = run(GrepTool(), {"pattern": "hello", "path": str(tmp_path)})
        assert "<pattern>hello</pattern>" in result
        assert "<path>" in result
        assert "<matches>" in result
        assert "</matches>" in result
        assert "<summary>" in result


class TestGrepToolIncludeFilter:
    """Test file type filtering."""

    def test_include_py_excludes_txt(self, tmp_path):
        (tmp_path / "code.py").write_text("target_string here\n")
        (tmp_path / "notes.txt").write_text("target_string here too\n")
        result = run(
            GrepTool(),
            {"pattern": "target_string", "path": str(tmp_path), "include": "*.py"},
        )
        assert "code.py" in result
        # The txt file should NOT appear in matches
        # (With rg it won't; with PythonEngine glob "*.py" only matches .py)
        lines_in_matches = result.split("<matches>\n")[1].split("\n</matches>")[0]
        assert "notes.txt" not in lines_in_matches


class TestGrepToolEmptyResults:
    """Test no-match scenarios."""

    def test_no_matches_returns_message(self, tmp_path):
        (tmp_path / "app.py").write_text("hello world\n")
        result = run(GrepTool(), {"pattern": "xyzzy_not_found", "path": str(tmp_path)})
        assert "No matches found" in result
        assert "<pattern>xyzzy_not_found</pattern>" in result


class TestGrepToolInvalidRegex:
    """Test error handling for bad patterns."""

    def test_invalid_regex_returns_error(self, tmp_path):
        (tmp_path / "app.py").write_text("some text\n")
        result = run(
            GrepTool(),
            {"pattern": "[invalid(", "path": str(tmp_path)},
        )
        assert "Error:" in result


class TestGrepToolMissingParameter:
    """Test required parameter validation."""

    def test_missing_pattern_returns_error(self):
        result = run(GrepTool(), {})
        assert "Error" in result

    def test_empty_pattern_returns_error(self):
        result = run(GrepTool(), {"pattern": ""})
        assert "Error" in result


class TestGrepToolDefaultPath:
    """Test default path behavior."""

    def test_default_path_is_workspace(self, tmp_path, monkeypatch):
        # Create files in tmp_path and set workspace to tmp_path
        (tmp_path / "unique_test_file.py").write_text("findme_default_path\n")
        from engine.config import Config
        config = Config(workspace=str(tmp_path))
        monkeypatch.setattr("engine.tools.builtin.grep.get_config", lambda: config)
        result = run(GrepTool(), {"pattern": "findme_default_path"})
        assert "unique_test_file.py" in result
        assert "<summary>" in result


class TestGrepToolSummary:
    """Test summary line details."""

    def test_summary_shows_match_count(self, tmp_path):
        (tmp_path / "app.py").write_text("foo\nbar foo\nbaz\n")
        result = run(GrepTool(), {"pattern": "foo", "path": str(tmp_path)})
        assert "2 matches" in result
        assert "1 file(s)" in result

    def test_path_is_absolute(self, tmp_path):
        (tmp_path / "app.py").write_text("hello\n")
        result = run(GrepTool(), {"pattern": "hello", "path": str(tmp_path)})
        assert str(tmp_path) in result
