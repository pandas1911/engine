"""Tests for engine/tools/builtin/_bash/output.py"""

from __future__ import annotations

from pathlib import Path

from engine.tools.builtin._bash.output import OutputBuffer, OutputTruncator


class TestOutputTruncator:
    def test_no_truncation_under_limits(self) -> None:
        truncator = OutputTruncator(max_lines=10, max_bytes=1024)
        result = truncator.truncate("hello\nworld")
        assert result.truncated is False
        assert result.output == "hello\nworld"
        assert result.full_output_path is None

    def test_truncation_at_line_limit(self) -> None:
        truncator = OutputTruncator(max_lines=5, max_bytes=1024 * 1024, output_dir=Path("/tmp/test-output"))
        lines = [f"line {i}" for i in range(100)]
        text = "\n".join(lines)
        result = truncator.truncate(text, persist=False)
        assert result.truncated is True
        assert "truncated" in result.output

    def test_truncation_at_byte_limit(self) -> None:
        truncator = OutputTruncator(max_lines=99999, max_bytes=10, output_dir=Path("/tmp/test-output"))
        text = "a" * 100
        result = truncator.truncate(text, persist=False)
        assert result.truncated is True
        assert "truncated" in result.output

    def test_tail_extraction(self) -> None:
        truncator = OutputTruncator(max_lines=5, max_bytes=1024 * 1024, output_dir=Path("/tmp/test-output"))
        lines = [f"line {i}" for i in range(100)]
        text = "\n".join(lines)
        result = truncator.truncate(text, persist=False)
        for i in range(95, 100):
            assert f"line {i}" in result.output

    def test_full_output_saved_to_file(self, tmp_path: Path) -> None:
        truncator = OutputTruncator(max_lines=3, max_bytes=1024 * 1024, output_dir=tmp_path / "out")
        lines = [f"line {i}" for i in range(50)]
        text = "\n".join(lines)
        result = truncator.truncate(text, persist=True)
        assert result.full_output_path is not None
        saved = Path(result.full_output_path)
        assert saved.exists()
        assert saved.read_text(encoding="utf-8") == text

    def test_no_persist_flag(self) -> None:
        truncator = OutputTruncator(max_lines=3, max_bytes=1024 * 1024, output_dir=Path("/tmp/test-output"))
        text = "\n".join(f"line {i}" for i in range(50))
        result = truncator.truncate(text, persist=False)
        assert result.full_output_path is None
        assert result.truncated is True


class TestOutputBuffer:
    def test_output_buffer_append(self) -> None:
        buf = OutputBuffer(max_bytes=1024)
        buf.append("hello ")
        buf.append("world")
        assert buf.get_text() == "hello world"

    def test_output_buffer_sliding_window(self) -> None:
        buf = OutputBuffer(max_bytes=10)
        buf.append("aaaa")  # 4 bytes
        buf.append("bbbb")  # 4 bytes, total 8
        buf.append("cccc")  # 4 bytes, total 12 > 10, discards "aaaa"
        text = buf.get_text()
        assert "aaaa" not in text
        assert "bbbb" in text
        assert "cccc" in text

    def test_output_buffer_empty(self) -> None:
        buf = OutputBuffer()
        assert buf.get_text() == ""
