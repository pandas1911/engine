"""Provider-specific thinking content extraction strategy.

Maps base_url domain to extraction logic. Each strategy is a class
that encapsulates its own state and extraction algorithm.
stream_chat() just calls extractor.extract(delta) per chunk.
"""
from __future__ import annotations
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


# ── Strategy: reasoning_details (MiniMax) ──

class ReasoningDetailsExtractor(ThinkingExtractor):
    """MiniMax: delta.reasoning_details[].text is CUMULATIVE.
    Must diff against buffer to get incremental text.
    """

    def __init__(self):
        self._buffer = ""

    def extract(self, delta: Any) -> ThinkingResult:
        thinking_text = ""
        if hasattr(delta, "reasoning_details") and delta.reasoning_details:
            for detail in delta.reasoning_details:
                if isinstance(detail, dict) and "text" in detail:
                    full_text = detail["text"]
                    new_text = full_text[len(self._buffer):]
                    if new_text:
                        thinking_text = new_text
                        self._buffer = full_text
        response_text = ""
        if hasattr(delta, "content") and delta.content:
            response_text = delta.content
        return ThinkingResult(
            thinking_text=thinking_text,
            response_text=response_text,
            source="reasoning_details" if thinking_text else None,
        )


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
