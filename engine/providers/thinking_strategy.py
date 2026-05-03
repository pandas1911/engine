"""Provider-specific thinking content extraction strategy.

Maps base_url domain to extraction logic. Each strategy is a class
that encapsulates its own state and extraction algorithm.
stream_chat() just calls extractor.extract(delta) per chunk.
"""
from __future__ import annotations
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse


# ── Extraction result ──

@dataclass
class ThinkingResult:
    """Returned by every extractor.extract() call."""
    thinking_text: str = ""
    response_text: str = ""
    source: Optional[str] = None


# ── Abstract base ──

class ThinkingExtractor(ABC):
    """Base class for provider-specific thinking extractors."""

    @abstractmethod
    def extract(self, delta: Any) -> ThinkingResult:
        """Extract thinking/response text from a streaming chunk delta."""
        ...

    @abstractmethod
    def flush(self) -> ThinkingResult:
        """Flush any remaining buffered content when stream ends."""
        ...


# ── Strategy: reasoning_details (MiniMax) ──

class ReasoningDetailsExtractor(ThinkingExtractor):
    """MiniMax: delta.reasoning_details[].text is incremental.
    Each chunk contains only new text — no diff or buffer needed.

    BUG HISTORY (for future reference):
      The original implementation used a diff algorithm
      (new_text = full_text[len(self._buffer):]) matching the official MiniMax
      docs, which state that reasoning_details[].text is cumulative. However,
      the actual MiniMax API returns INCREMENTAL text — each chunk contains
      only the new characters since the last chunk.

      This caused all thinking content after the first chunk to be silently
      dropped: self._buffer grew to the length of the first chunk's text, and
      subsequent incremental texts were shorter than self._buffer, so the slice
      full_text[len(self._buffer):] returned "" every time.

      Evidence: https://github.com/MiniMax-AI/MiniMax-M2/issues/95

    Additionally, MiniMax sometimes leaks thinking content into the response
    content field prefixed with "(think)\\n". We strip that here.
      See: https://github.com/MiniMax-AI/MiniMax-M2/issues/105
    """

    def extract(self, delta: Any) -> ThinkingResult:
        # Accumulate incremental text from each reasoning_details entry.
        # No state tracking needed — each detail["text"] is only the new text.
        thinking_text = ""
        if hasattr(delta, "reasoning_details") and delta.reasoning_details:
            for detail in delta.reasoning_details:
                # Support both dict objects and SDK objects with a .text attr.
                # Some MiniMax SDK versions return non-dict objects for details.
                if isinstance(detail, dict) and "text" in detail:
                    thinking_text += detail["text"]
                elif hasattr(detail, "text"):
                    thinking_text += detail.text

        response_text = ""
        if hasattr(delta, "content") and delta.content:
            response_text = delta.content

        # Strip the "(think)\n" prefix that MiniMax sometimes leaks into
        # the content field when thinking content bleeds through.
        if response_text:
            response_text = re.sub(r"^\(think\)\s*\n?", "", response_text)

        return ThinkingResult(
            thinking_text=thinking_text,
            response_text=response_text,
            source="reasoning_details" if thinking_text else None,
        )

    def flush(self) -> ThinkingResult:
        return ThinkingResult()


# ── Strategy: reasoning_content (Qwen/DashScope) ──

class ReasoningContentExtractor(ThinkingExtractor):
    """Qwen/DashScope: delta.reasoning_content is incremental text.
    No state tracking needed.
    """

    def extract(self, delta: Any) -> ThinkingResult:
        thinking_text = ""
        if (hasattr(delta, "reasoning_content")
                and delta.reasoning_content is not None):
            thinking_text = delta.reasoning_content
        response_text = ""
        if hasattr(delta, "content") and delta.content:
            response_text = delta.content
        return ThinkingResult(
            thinking_text=thinking_text,
            response_text=response_text,
            source="reasoning_content" if thinking_text else None,
        )

    def flush(self) -> ThinkingResult:
        return ThinkingResult()


# ── Strategy: tag_parser (default/DeepSeek) ──

class TagParserExtractor(ThinkingExtractor):
    """Default/DeepSeek: <think/> tags in delta.content.
    Uses ThinkingCapture state machine internally.
    """

    def __init__(self):
        from engine.providers.thinking_capture import ThinkingCapture
        self._capture = ThinkingCapture()

    def extract(self, delta: Any) -> ThinkingResult:
        if not (hasattr(delta, "content") and delta.content):
            return ThinkingResult()
        parsed = self._capture.feed(delta.content)
        return ThinkingResult(
            thinking_text=parsed.thinking_text,
            response_text=parsed.response_text,
            source="tag_parser" if parsed.thinking_text else None,
        )

    def flush(self) -> ThinkingResult:
        parsed = self._capture.flush()
        return ThinkingResult(
            thinking_text=parsed.thinking_text,
            response_text=parsed.response_text,
            source="tag_parser" if parsed.thinking_text else None,
        )


# ── Registry and factory ──

_DOMAIN_EXTRACTORS: dict[str, type[ThinkingExtractor]] = {
    "api.minimaxi.com":       ReasoningDetailsExtractor,
    "dashscope.aliyuncs.com": ReasoningContentExtractor,
}

_DEFAULT_EXTRACTOR = TagParserExtractor


def get_thinking_extractor(base_url: str) -> ThinkingExtractor:
    """Create the appropriate thinking extractor for a provider's base_url.

    Args:
        base_url: The provider's API base URL (e.g., "https://api.minimaxi.com/v1")

    Returns:
        A ThinkingExtractor instance ready to use.
    """
    try:
        hostname = urlparse(base_url).hostname or ""
    except Exception:
        return _DEFAULT_EXTRACTOR()
    extractor_cls = _DOMAIN_EXTRACTORS.get(hostname, _DEFAULT_EXTRACTOR)
    return extractor_cls()
