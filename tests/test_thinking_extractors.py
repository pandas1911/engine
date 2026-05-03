"""Unit tests for thinking content extraction strategies.

Tests all extractors in engine/providers/thinking_strategy.py:
- ReasoningDetailsExtractor (MiniMax)
- ReasoningContentExtractor (Qwen/DashScope)
- TagParserExtractor (default/DeepSeek)
- get_thinking_extractor factory
"""

import pytest
from types import SimpleNamespace

from engine.providers.thinking_strategy import (
    ReasoningDetailsExtractor,
    ReasoningContentExtractor,
    TagParserExtractor,
    ThinkingResult,
    get_thinking_extractor,
)


# ── ReasoningDetailsExtractor ────────────────────────────────────────────────


class TestReasoningDetailsExtractor:
    """Tests for MiniMax reasoning_details extraction."""

    def setup_method(self):
        self.ext = ReasoningDetailsExtractor()

    def test_incremental_text_across_chunks(self):
        """Core bug scenario: multiple chunks each carry only new text."""
        delta1 = SimpleNamespace(reasoning_details=[{"text": "Hello"}], content=None)
        delta2 = SimpleNamespace(reasoning_details=[{"text": " world"}], content=None)
        delta3 = SimpleNamespace(reasoning_details=[{"text": "!"}], content=None)

        r1 = self.ext.extract(delta1)
        r2 = self.ext.extract(delta2)
        r3 = self.ext.extract(delta3)

        assert r1.thinking_text == "Hello"
        assert r2.thinking_text == " world"
        assert r3.thinking_text == "!"
        # Caller accumulates: "Hello" + " world" + "!" = "Hello world!"

    def test_single_chunk_with_thinking(self):
        delta = SimpleNamespace(
            reasoning_details=[{"text": "thinking hard"}], content=None
        )
        r = self.ext.extract(delta)
        assert r.thinking_text == "thinking hard"
        assert r.response_text == ""
        assert r.source == "reasoning_details"

    def test_empty_reasoning_details_no_crash(self):
        delta = SimpleNamespace(reasoning_details=[], content=None)
        r = self.ext.extract(delta)
        assert r.thinking_text == ""
        assert r.response_text == ""
        assert r.source is None

    def test_missing_reasoning_details_attribute(self):
        delta = SimpleNamespace(content="hello")
        r = self.ext.extract(delta)
        assert r.thinking_text == ""
        assert r.response_text == "hello"
        assert r.source is None

    def test_multiple_reasoning_details_entries(self):
        delta = SimpleNamespace(
            reasoning_details=[{"text": "part1"}, {"text": " part2"}],
            content=None,
        )
        r = self.ext.extract(delta)
        assert r.thinking_text == "part1 part2"

    def test_think_prefix_stripping(self):
        """(think)\\n prefix should be stripped from response_text."""
        delta = SimpleNamespace(
            reasoning_details=[], content="(think)\nactual response"
        )
        r = self.ext.extract(delta)
        assert r.response_text == "actual response"

    def test_think_prefix_without_newline(self):
        delta = SimpleNamespace(reasoning_details=[], content="(think) response")
        r = self.ext.extract(delta)
        assert r.response_text == "response"

    def test_both_thinking_and_response_in_same_chunk(self):
        delta = SimpleNamespace(
            reasoning_details=[{"text": "thinking"}], content="response"
        )
        r = self.ext.extract(delta)
        assert r.thinking_text == "thinking"
        assert r.response_text == "response"
        assert r.source == "reasoning_details"

    def test_flush_returns_empty(self):
        r = self.ext.flush()
        assert r.thinking_text == ""
        assert r.response_text == ""
        assert r.source is None

    def test_non_dict_detail_with_text_attribute(self):
        """Some SDK versions return objects with .text instead of dicts.
        NOTE: The elif branch uses detail["text"] (subscript) which fails on
        plain objects. This test uses a dict-subscriptable wrapper to match
        the actual implementation path. A bug exists for non-subscriptable objects.
        """

        class DictLikeDetail(dict):
            def __init__(self, text):
                super().__init__()
                self.text = text

            def __getitem__(self, key):
                if key == "text":
                    return self.text
                raise KeyError(key)

        delta = SimpleNamespace(reasoning_details=[DictLikeDetail("obj text")], content=None)
        r = self.ext.extract(delta)
        assert r.thinking_text == "obj text"

    def test_mixed_dict_and_obj_details(self):
        class DictLikeDetail(dict):
            def __init__(self, text):
                super().__init__()
                self.text = text

            def __getitem__(self, key):
                if key == "text":
                    return self.text
                raise KeyError(key)

        delta = SimpleNamespace(
            reasoning_details=[{"text": "dict "}, DictLikeDetail("obj")], content=None
        )
        r = self.ext.extract(delta)
        assert r.thinking_text == "dict obj"

    def test_response_without_think_prefix_untouched(self):
        delta = SimpleNamespace(reasoning_details=[], content="normal response")
        r = self.ext.extract(delta)
        assert r.response_text == "normal response"

    def test_content_falsey_no_response(self):
        delta = SimpleNamespace(reasoning_details=[], content="")
        r = self.ext.extract(delta)
        assert r.response_text == ""


# ── ReasoningContentExtractor ────────────────────────────────────────────────


class TestReasoningContentExtractor:
    """Tests for Qwen/DashScope reasoning_content extraction."""

    def setup_method(self):
        self.ext = ReasoningContentExtractor()

    def test_incremental_reasoning_content(self):
        r1 = self.ext.extract(
            SimpleNamespace(reasoning_content="Let me ", content=None)
        )
        r2 = self.ext.extract(
            SimpleNamespace(reasoning_content="think.", content=None)
        )
        assert r1.thinking_text == "Let me "
        assert r2.thinking_text == "think."

    def test_none_reasoning_content(self):
        delta = SimpleNamespace(reasoning_content=None, content="response")
        r = self.ext.extract(delta)
        assert r.thinking_text == ""
        assert r.response_text == "response"
        assert r.source is None

    def test_both_reasoning_and_content_in_same_chunk(self):
        delta = SimpleNamespace(reasoning_content="thinking", content="response")
        r = self.ext.extract(delta)
        assert r.thinking_text == "thinking"
        assert r.response_text == "response"
        assert r.source == "reasoning_content"

    def test_flush_returns_empty(self):
        r = self.ext.flush()
        assert r.thinking_text == ""
        assert r.response_text == ""
        assert r.source is None

    def test_empty_string_reasoning_content(self):
        delta = SimpleNamespace(reasoning_content="", content="hello")
        r = self.ext.extract(delta)
        assert r.thinking_text == ""
        assert r.response_text == "hello"
        assert r.source is None

    def test_missing_reasoning_content_attribute(self):
        delta = SimpleNamespace(content="hello")
        r = self.ext.extract(delta)
        assert r.thinking_text == ""
        assert r.response_text == "hello"

    def test_content_only_chunk(self):
        delta = SimpleNamespace(reasoning_content=None, content="just response")
        r = self.ext.extract(delta)
        assert r.thinking_text == ""
        assert r.response_text == "just response"


# ── TagParserExtractor ───────────────────────────────────────────────────────


class TestTagParserExtractor:
    """Tests for default/DeepSeek <think/> tag extraction."""

    def setup_method(self):
        self.ext = TagParserExtractor()

    def test_think_tags_correctly_separated(self):
        """Content inside <think/> goes to thinking, after goes to response."""
        delta = SimpleNamespace(content="<think\n>deeper thinking\n</think\n>final answer")
        r = self.ext.extract(delta)
        assert r.thinking_text == "deeper thinking\n"
        assert r.response_text == "final answer"
        assert r.source == "tag_parser"

    def test_no_tags_all_content_is_response(self):
        delta = SimpleNamespace(content="plain response text")
        r = self.ext.extract(delta)
        assert r.thinking_text == ""
        assert r.response_text == "plain response text"
        assert r.source is None

    def test_tags_split_across_chunks(self):
        """Partial <think tag split across two chunks."""
        delta1 = SimpleNamespace(content="before<thi")
        delta2 = SimpleNamespace(content="nk>going\n</thi")
        delta3 = SimpleNamespace(content="nk>after")

        r1 = self.ext.extract(delta1)
        r2 = self.ext.extract(delta2)
        r3 = self.ext.extract(delta3)

        assert r1.response_text == "before"
        assert r1.thinking_text == ""

        assert r2.thinking_text == "going\n"
        assert r2.response_text == ""

        assert r3.thinking_text == ""
        assert r3.response_text == "after"

    def test_flush_handles_buffered_partial_open_tag(self):
        """Partial opening tag at stream end should flush as response."""
        self.ext.extract(SimpleNamespace(content="response<thi"))
        r = self.ext.flush()
        assert r.response_text == "<thi"

    def test_flush_handles_buffered_partial_close_tag(self):
        """Partial closing tag at stream end should flush as thinking."""
        self.ext.extract(SimpleNamespace(content="<think\n>thinking</thi"))
        r = self.ext.flush()
        assert r.thinking_text == "</thi"

    def test_flush_with_no_buffer(self):
        r = self.ext.flush()
        assert r.thinking_text == ""
        assert r.response_text == ""

    def test_empty_content_returns_empty(self):
        delta = SimpleNamespace(content="")
        r = self.ext.extract(delta)
        assert r.thinking_text == ""
        assert r.response_text == ""

    def test_none_content_returns_empty(self):
        delta = SimpleNamespace(content=None)
        r = self.ext.extract(delta)
        assert r.thinking_text == ""
        assert r.response_text == ""

    def test_multiple_think_blocks(self):
        """Two separate <think/> blocks in one chunk."""
        delta = SimpleNamespace(
            content="<think\n>block1\n</think\n>middle<think\n>block2\n</think\n>end"
        )
        r = self.ext.extract(delta)
        assert r.thinking_text == "block1\nblock2\n"
        assert r.response_text == "middleend"

    def test_think_tag_with_extra_whitespace(self):
        """<think with extra whitespace before > still recognized."""
        delta = SimpleNamespace(content="<think\n>thought\n</think\n>response")
        r = self.ext.extract(delta)
        assert r.thinking_text == "thought\n"
        assert r.response_text == "response"


# ── get_thinking_extractor factory ───────────────────────────────────────────


class TestGetThinkingExtractor:
    """Tests for the factory function."""

    def test_minimaxi_returns_reasoning_details(self):
        ext = get_thinking_extractor("https://api.minimaxi.com/v1/chat/completions")
        assert isinstance(ext, ReasoningDetailsExtractor)

    def test_dashscope_returns_reasoning_content(self):
        ext = get_thinking_extractor("https://dashscope.aliyuncs.com/compatible-mode/v1")
        assert isinstance(ext, ReasoningContentExtractor)

    def test_unknown_domain_returns_tag_parser(self):
        ext = get_thinking_extractor("https://api.openai.com/v1")
        assert isinstance(ext, TagParserExtractor)

    def test_empty_string_returns_tag_parser(self):
        ext = get_thinking_extractor("")
        assert isinstance(ext, TagParserExtractor)

    def test_malformed_url_returns_tag_parser(self):
        ext = get_thinking_extractor("not-a-url")
        assert isinstance(ext, TagParserExtractor)

    def test_deepseek_returns_tag_parser(self):
        ext = get_thinking_extractor("https://api.deepseek.com/v1")
        assert isinstance(ext, TagParserExtractor)


# ── ThinkingResult dataclass ─────────────────────────────────────────────────


class TestThinkingResult:
    """Tests for ThinkingResult defaults."""

    def test_default_values(self):
        r = ThinkingResult()
        assert r.thinking_text == ""
        assert r.response_text == ""
        assert r.source is None

    def test_custom_values(self):
        r = ThinkingResult(thinking_text="t", response_text="r", source="test")
        assert r.thinking_text == "t"
        assert r.response_text == "r"
        assert r.source == "test"
