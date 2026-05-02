"""Tag-based thinking content capture for streaming (tag_parser strategy).

Only used when thinking_strategy is "tag_parser" (default/DeepSeek).
Other strategies (reasoning_content, reasoning_details) handle
extraction directly in stream_chat() — no tag parsing needed.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ThinkState(Enum):
    OUTSIDE = "outside"
    INSIDE = "inside"
    MAYBE_OPEN = "maybe_open"    # Partial "<thi..."
    MAYBE_CLOSE = "maybe_close"  # Partial "</thi..."


@dataclass
class CaptureResult:
    """Result from processing a single content chunk."""
    thinking_text: str = ""
    response_text: str = ""


# Opening/closing tag prefixes for incremental detection
_OPEN_TAG = "<think"
_CLOSE_TAG = "</think"
_TAG_END_CHARS = {">", " ", "\t", "\n", "\r"}


class ThinkingCapture:
    """Stateful thinking tag parser for tag_parser strategy (DeepSeek/default).

    Handles partial tags split across streaming chunks.
    Only used when thinking_strategy is "tag_parser".
    """

    def __init__(self):
        self.state = ThinkState.OUTSIDE
        self._buffer: str = ""       # Accumulates partial tag text

    def feed(self, content: str) -> CaptureResult:
        """Process a content chunk. Returns separated thinking/response text."""
        thinking_parts: list[str] = []
        response_parts: list[str] = []

        i = 0
        while i < len(content):
            ch = content[i]

            if self.state == ThinkState.OUTSIDE:
                if ch == "<":
                    remaining = content[i:]
                    if remaining.startswith(_OPEN_TAG):
                        end = self._find_tag_end(remaining, len(_OPEN_TAG))
                        if end is not None:
                            self.state = ThinkState.INSIDE
                            i += end + 1
                            continue
                        else:
                            self._buffer = remaining
                            self.state = ThinkState.MAYBE_OPEN
                            break
                    elif _OPEN_TAG.startswith(remaining):
                        self._buffer = remaining
                        self.state = ThinkState.MAYBE_OPEN
                        break
                response_parts.append(ch)
                i += 1

            elif self.state == ThinkState.INSIDE:
                if ch == "<":
                    remaining = content[i:]
                    if remaining.startswith(_CLOSE_TAG):
                        end = self._find_tag_end(remaining, len(_CLOSE_TAG))
                        if end is not None:
                            self.state = ThinkState.OUTSIDE
                            i += end + 1
                            continue
                        else:
                            self._buffer = remaining
                            self.state = ThinkState.MAYBE_CLOSE
                            break
                    elif _CLOSE_TAG.startswith(remaining):
                        self._buffer = remaining
                        self.state = ThinkState.MAYBE_CLOSE
                        break
                thinking_parts.append(ch)
                i += 1

            elif self.state == ThinkState.MAYBE_OPEN:
                self._buffer += ch
                i += 1
                if self._buffer.startswith(_OPEN_TAG):
                    end = self._find_tag_end(self._buffer, len(_OPEN_TAG))
                    if end is not None:
                        self.state = ThinkState.INSIDE
                        self._buffer = ""
                        continue
                elif not _OPEN_TAG.startswith(self._buffer):
                    response_parts.append(self._buffer)
                    self._buffer = ""
                    self.state = ThinkState.OUTSIDE

            elif self.state == ThinkState.MAYBE_CLOSE:
                self._buffer += ch
                i += 1
                if self._buffer.startswith(_CLOSE_TAG):
                    end = self._find_tag_end(self._buffer, len(_CLOSE_TAG))
                    if end is not None:
                        self.state = ThinkState.OUTSIDE
                        self._buffer = ""
                        continue
                elif not _CLOSE_TAG.startswith(self._buffer):
                    thinking_parts.append(self._buffer)
                    self._buffer = ""
                    self.state = ThinkState.INSIDE

        return CaptureResult(
            thinking_text="".join(thinking_parts),
            response_text="".join(response_parts),
        )

    def _find_tag_end(self, text: str, start: int) -> Optional[int]:
        """Find the position of '>' after a tag prefix. Returns index or None."""
        for i in range(start, len(text)):
            if text[i] == ">":
                return i
            if text[i] in (" ", "\t", "\n", "\r"):
                continue
            if not text[i].isalpha():
                return None  # Invalid char in tag
        return None  # Tag not closed yet
