"""Tests for ReadTool — file and directory reading with pagination, security, and binary detection."""

import asyncio
import os
import stat

import pytest

from engine.tools.builtin.read import ReadTool
from engine.tools.builtin.security import PathGuard


def _run(tool, args, context=None):
    return asyncio.run(tool.execute(args, context or {}))


class TestReadToolFileRead:
    """Tests for normal file reading with line numbers and XML formatting."""

    def test_read_text_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\n")
        result = _run(ReadTool(), {"filePath": str(f)})
        assert "<path>" in result
        assert "<type>file</type>" in result
        assert "<content>" in result
        assert "1: line1" in result
        assert "2: line2" in result
        assert "3: line3" in result

    def test_read_file_preserves_path(self, tmp_path):
        f = tmp_path / "hello.py"
        f.write_text("print('hi')\n")
        result = _run(ReadTool(), {"filePath": str(f)})
        assert str(f) in result

    def test_read_file_no_trailing_newline(self, tmp_path):
        f = tmp_path / "no_newline.txt"
        f.write_text("single line")
        result = _run(ReadTool(), {"filePath": str(f)})
        assert "1: single line" in result


class TestReadToolDirectory:
    """Tests for directory listing with / suffix for subdirectories."""

    def test_directory_listing(self, tmp_path):
        (tmp_path / "file_a.txt").write_text("a")
        (tmp_path / "file_b.py").write_text("b")
        sub = tmp_path / "subdir"
        sub.mkdir()
        result = _run(ReadTool(), {"filePath": str(tmp_path)})
        assert "<type>directory</type>" in result
        assert "<entries>" in result
        assert "subdir/" in result
        assert "file_a.txt" in result
        assert "file_b.py" in result

    def test_empty_directory(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = _run(ReadTool(), {"filePath": str(empty_dir)})
        assert "<type>directory</type>" in result
        assert "<entries>" in result
        assert str(empty_dir) in result


class TestReadToolPagination:
    """Tests for offset/limit pagination."""

    def test_offset_limit(self, tmp_path):
        lines = [f"line{i}" for i in range(1, 11)]
        f = tmp_path / "paged.txt"
        f.write_text("\n".join(lines))
        result = _run(ReadTool(), {"filePath": str(f), "offset": 3, "limit": 3})
        assert "3: line3" in result
        assert "4: line4" in result
        assert "5: line5" in result
        assert "2: line2" not in result
        assert "6: line6" not in result

    def test_offset_only(self, tmp_path):
        lines = [f"row{i}" for i in range(1, 6)]
        f = tmp_path / "offset.txt"
        f.write_text("\n".join(lines))
        result = _run(ReadTool(), {"filePath": str(f), "offset": 4})
        assert "4: row4" in result
        assert "5: row5" in result
        assert "3: row3" not in result

    def test_limit_only(self, tmp_path):
        lines = [f"v{i}" for i in range(1, 20)]
        f = tmp_path / "limited.txt"
        f.write_text("\n".join(lines))
        result = _run(ReadTool(), {"filePath": str(f), "limit": 3})
        assert "1: v1" in result
        assert "2: v2" in result
        assert "3: v3" in result
        assert "4: v4" not in result


class TestReadToolNotFound:
    """Tests for file-not-found error with sibling suggestions."""

    def test_file_not_found_with_siblings(self, tmp_path):
        (tmp_path / "sibling_a.txt").write_text("a")
        (tmp_path / "sibling_b.txt").write_text("b")
        missing = tmp_path / "nonexistent.txt"
        result = _run(ReadTool(), {"filePath": str(missing)})
        assert "Error" in result
        assert "not found" in result.lower() or "not found" in result
        assert "sibling_a.txt" in result or "Sibling" in result

    def test_file_not_found_empty_dir(self, tmp_path):
        missing = tmp_path / "ghost.txt"
        result = _run(ReadTool(), {"filePath": str(missing)})
        assert "Error" in result


class TestReadToolBinary:
    """Tests for binary file rejection."""

    def test_binary_exe_rejected(self, tmp_path):
        f = tmp_path / "program.exe"
        f.write_bytes(b"\x00\x01\x02\x03")
        result = _run(ReadTool(), {"filePath": str(f)})
        assert "binary" in result.lower()
        assert "Error" in result


class TestReadToolTruncation:
    """Tests for large file and long line truncation."""

    def test_large_file_truncation(self, tmp_path):
        lines = [f"line {i} content here" for i in range(1, 2100)]
        f = tmp_path / "large.txt"
        f.write_text("\n".join(lines))
        result = _run(ReadTool(), {"filePath": str(f), "limit": 2000})
        # Should contain truncation hint with grep or sub-agent
        assert "grep" in result.lower() or "sub-agent" in result.lower()

    def test_long_line_truncation(self, tmp_path):
        long_line = "x" * 3000
        f = tmp_path / "longline.txt"
        f.write_text(long_line)
        result = _run(ReadTool(), {"filePath": str(f)})
        assert "[truncated]" in result


class TestReadToolSecurity:
    """Tests for path security deny-list."""

    def test_env_file_denied(self, tmp_path):
        # Use a PathGuard with custom pattern to ensure test reliability
        f = tmp_path / ".env"
        f.write_text("SECRET=123")
        guard = PathGuard(denied_patterns=["**/.env"])
        result = _run(ReadTool(path_guard=guard), {"filePath": str(f)})
        assert "denied pattern" in result
        assert "Error" in result

    def test_credentials_denied(self, tmp_path):
        f = tmp_path / "credentials.json"
        f.write_text('{"key": "secret"}')
        guard = PathGuard(denied_patterns=["**/credentials*"])
        result = _run(ReadTool(path_guard=guard), {"filePath": str(f)})
        assert "denied pattern" in result


class TestReadToolEmptyFile:
    """Tests for empty file handling."""

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        result = _run(ReadTool(), {"filePath": str(f)})
        assert "<type>file</type>" in result
        assert "<content>" in result
        assert str(f) in result


class TestReadToolMissingParameter:
    """Tests for missing or invalid filePath parameter."""

    def test_missing_file_path(self):
        result = _run(ReadTool(), {})
        assert "Error" in result
        assert "required" in result.lower()

    def test_empty_file_path(self):
        result = _run(ReadTool(), {"filePath": ""})
        assert "Error" in result

    def test_none_file_path(self):
        result = _run(ReadTool(), {"filePath": None})
        assert "Error" in result


class TestReadToolPermissionDenied:
    """Tests for permission denied handling."""

    def test_permission_denied_file(self, tmp_path):
        f = tmp_path / "noperm.txt"
        f.write_text("secret stuff")
        # Remove read permission
        os.chmod(str(f), 0o000)
        try:
            result = _run(ReadTool(), {"filePath": str(f)})
            assert "Error" in result
            assert "Permission denied" in result or "permission" in result.lower()
        finally:
            # Restore permissions for cleanup
            os.chmod(str(f), stat.S_IRUSR | stat.S_IWUSR)

    def test_permission_denied_directory(self, tmp_path):
        d = tmp_path / "noperm_dir"
        d.mkdir()
        (d / "inner.txt").write_text("hidden")
        # Remove read+execute permission on directory
        os.chmod(str(d), 0o000)
        try:
            result = _run(ReadTool(), {"filePath": str(d)})
            assert "Error" in result
        finally:
            os.chmod(str(d), stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
