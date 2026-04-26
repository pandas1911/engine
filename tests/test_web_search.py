"""Integration tests for WebSearchTool (DuckDuckGo HTML endpoint).

These tests perform real network requests against DuckDuckGo.
They may fail due to rate limiting or network issues.
"""

import pytest

from engine.tools.custom.web_search import WebSearchTool


# live integration test
@pytest.mark.asyncio
@pytest.mark.integration
async def test_search_basic():
    """Basic search returns formatted results (or graceful error if rate-limited)."""
    tool = WebSearchTool()
    result = await tool.execute({"query": "Python programming"}, {})
    assert isinstance(result, str)
    assert len(result) > 0
    # Successful search should have numbered entries and URLs
    # Rate-limited search should return a graceful error string
    if not result.startswith("Web search error:"):
        assert "[1]" in result or "1." in result, "Expected numbered entries"
        assert "http" in result, "Expected URLs in results"


# live integration test
@pytest.mark.asyncio
@pytest.mark.integration
async def test_search_empty_query():
    """Empty query returns an error message."""
    tool = WebSearchTool()
    result = await tool.execute({"query": ""}, {})
    assert "error" in result.lower() or "empty" in result.lower()


# live integration test
@pytest.mark.asyncio
@pytest.mark.integration
async def test_search_special_characters():
    """Query with special characters returns valid results."""
    tool = WebSearchTool()
    result = await tool.execute({"query": "C++ STL containers"}, {})
    assert isinstance(result, str)
    assert len(result) > 0


def test_tool_schema():
    """Tool metadata and parameter schema are correct."""
    tool = WebSearchTool()
    assert tool.name == "web_search"
    props = tool.parameters["properties"]
    assert "query" in props
    assert "freshness" in props
    assert tool.parameters["required"] == ["query"]
    # Internal config keys must not leak into the public schema
    assert "max_results" not in props
    assert "region" not in props
    assert "safe_search" not in props
    # Description should not expose implementation details
    assert "duckduckgo" not in tool.description.lower()
