"""Tests for SearchEngine — ripgrep/Python search abstraction layer."""

import os
import shutil

import pytest

from engine.tools.builtin.search import (
    PythonEngine,
    RipgrepEngine,
    SearchEngine,
    SearchResult,
    get_search_engine,
    MAX_RESULTS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def search_dir(tmp_path):
    """Create a temporary directory with test files for searching."""
    # Python file with functions
    (tmp_path / "app.py").write_text("def hello():\n    return 'hello'\n\ndef world():\n    return 'world'\n")
    # Another Python file
    (tmp_path / "utils.py").write_text("def helper():\n    pass\n\n# A comment\n")
    # Text file
    (tmp_path / "readme.txt").write_text("Hello World\nThis is a test file\nSearch me!\n")
    # Subdirectory with files
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "module.py").write_text("def nested_func():\n    return True\n")
    (sub / "data.txt").write_text("Nested data here\n")
    return tmp_path


# ---------------------------------------------------------------------------
# SearchResult dataclass
# ---------------------------------------------------------------------------

class TestSearchResult:
    def test_search_result_fields(self):
        r = SearchResult(file="/path/to/file.py", line_number=42, content="match here")
        assert r.file == "/path/to/file.py"
        assert r.line_number == 42
        assert r.content == "match here"


# ---------------------------------------------------------------------------
# PythonEngine — content search
# ---------------------------------------------------------------------------

class TestPythonEngineContentSearch:
    """Tests for PythonEngine.search_content()."""

    def test_basic_content_search(self, search_dir):
        engine = PythonEngine()
        results = engine.search_content("def hello", str(search_dir))
        assert len(results) >= 1
        assert any("app.py" in r.file for r in results)

    def test_search_returns_line_numbers(self, search_dir):
        engine = PythonEngine()
        results = engine.search_content("def hello", str(search_dir))
        assert len(results) >= 1
        assert results[0].line_number == 1

    def test_search_multiple_files(self, search_dir):
        engine = PythonEngine()
        results = engine.search_content("def ", str(search_dir))
        # Should find matches in app.py, utils.py, and subdir/module.py
        files = set(r.file for r in results)
        assert len(files) >= 3

    def test_search_with_include_filter(self, search_dir):
        engine = PythonEngine()
        results = engine.search_content("Hello", str(search_dir), include="*.txt")
        # Should only match in .txt files, not .py
        for r in results:
            assert r.file.endswith(".txt")

    def test_search_empty_results(self, search_dir):
        engine = PythonEngine()
        results = engine.search_content("xyzzy_not_found_anywhere", str(search_dir))
        assert results == []

    def test_search_invalid_regex(self, search_dir):
        engine = PythonEngine()
        with pytest.raises(ValueError, match="Invalid regex"):
            engine.search_content("[invalid(", str(search_dir))

    def test_search_result_limit(self, tmp_path):
        # Create a file with many matching lines
        f = tmp_path / "big.py"
        lines = [f"def func_{i}():" for i in range(200)]
        f.write_text("\n".join(lines))
        engine = PythonEngine()
        results = engine.search_content("def func_", str(f))
        assert len(results) <= MAX_RESULTS


# ---------------------------------------------------------------------------
# PythonEngine — file search
# ---------------------------------------------------------------------------

class TestPythonEngineFileSearch:
    """Tests for PythonEngine.search_files()."""

    def test_glob_py_files(self, search_dir):
        engine = PythonEngine()
        files = engine.search_files("*.py", str(search_dir))
        assert len(files) >= 2  # app.py, utils.py at minimum
        for f in files:
            assert f.endswith(".py")

    def test_glob_recursive(self, search_dir):
        engine = PythonEngine()
        files = engine.search_files("**/*.py", str(search_dir))
        # Should find app.py, utils.py, subdir/module.py
        assert len(files) >= 3

    def test_glob_no_match(self, search_dir):
        engine = PythonEngine()
        files = engine.search_files("*.xyz", str(search_dir))
        assert files == []


# ---------------------------------------------------------------------------
# RipgrepEngine — mocked tests
# ---------------------------------------------------------------------------

class TestRipgrepEngine:
    """Tests for RipgrepEngine (may skip if rg not installed)."""

    @pytest.fixture(autouse=True)
    def check_rg(self):
        if not shutil.which("rg"):
            pytest.skip("ripgrep not installed")

    def test_content_search_with_rg(self, search_dir):
        engine = RipgrepEngine()
        results = engine.search_content("def hello", str(search_dir))
        assert len(results) >= 1
        assert any("app.py" in r.file for r in results)

    def test_file_search_with_rg(self, search_dir):
        engine = RipgrepEngine()
        files = engine.search_files("*.py", str(search_dir))
        assert len(files) >= 1


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

class TestGetSearchEngine:
    """Tests for get_search_engine() factory."""

    def test_returns_search_engine(self):
        engine = get_search_engine()
        assert isinstance(engine, SearchEngine)

    def test_returns_correct_type(self):
        engine = get_search_engine()
        if shutil.which("rg"):
            assert isinstance(engine, RipgrepEngine)
        else:
            assert isinstance(engine, PythonEngine)
