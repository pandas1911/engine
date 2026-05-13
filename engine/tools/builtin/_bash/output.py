"""Output truncation with file persistence for large command output."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from engine.tools.builtin._bash.schemas import MAX_OUTPUT_BYTES, MAX_OUTPUT_LINES
except ImportError:
    MAX_OUTPUT_LINES = 2000
    MAX_OUTPUT_BYTES = 50 * 1024


@dataclass
class TruncationResult:
    output: str
    truncated: bool
    full_output_path: Optional[str] = None


class OutputTruncator:
    def __init__(
        self,
        max_lines: int = MAX_OUTPUT_LINES,
        max_bytes: int = MAX_OUTPUT_BYTES,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.max_lines = max_lines
        self.max_bytes = max_bytes
        self.output_dir = output_dir or Path(".engine/tool-output")

    def truncate(self, output: str, persist: bool = True) -> TruncationResult:
        lines = output.split("\n")
        if len(lines) <= self.max_lines and len(output.encode("utf-8")) <= self.max_bytes:
            return TruncationResult(output=output, truncated=False)
        tail_text = self._extract_tail(output)
        full_output_path = self._save_full_output(output) if persist else None
        if full_output_path:
            hint = f"\n\n... output truncated ... Full output saved to: {full_output_path}\n\n"
        else:
            hint = "\n\n... output truncated ...\n\n"
        return TruncationResult(
            output=hint + tail_text, truncated=True, full_output_path=full_output_path
        )

    def _extract_tail(self, output: str) -> str:
        lines = output.split("\n")
        selected: list[str] = []
        byte_count = 0
        for line in reversed(lines):
            line_bytes = len(line.encode("utf-8")) + (1 if selected else 0)
            if byte_count + line_bytes > self.max_bytes or len(selected) >= self.max_lines:
                break
            selected.insert(0, line)
            byte_count += line_bytes
        return "\n".join(selected)

    def _save_full_output(self, output: str) -> str:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.output_dir / f"bash-output-{uuid.uuid4().hex[:12]}.txt"
        filepath.write_text(output, encoding="utf-8")
        return str(filepath)


class OutputBuffer:
    """Sliding window buffer for streaming output capture."""

    def __init__(self, max_bytes: int = MAX_OUTPUT_BYTES * 2) -> None:
        self._max_bytes = max_bytes
        self._chunks: list[tuple[str, int]] = []
        self._total_bytes: int = 0

    def append(self, chunk: str) -> None:
        size = len(chunk.encode("utf-8"))
        self._chunks.append((chunk, size))
        self._total_bytes += size
        while self._total_bytes > self._max_bytes and len(self._chunks) > 1:
            _, old_size = self._chunks.pop(0)
            self._total_bytes -= old_size

    def get_text(self) -> str:
        return "".join(text for text, _ in self._chunks)
