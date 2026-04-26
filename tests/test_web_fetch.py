"""Unit tests for WebFetchTool using httpx.MockTransport — zero real network access."""

import asyncio

import httpx
import pytest

import engine.tools.custom.web_fetch as wf_module
from engine.runner import _discover_custom_tools
from engine.tools.custom.web_fetch import WebFetchTool

tool = WebFetchTool()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(handler) -> httpx.AsyncClient:
    """Build an AsyncClient backed by MockTransport for *handler*."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")


def _set_shared_client(handler):
    """Replace the module-level _shared_client with a mocked one."""
    wf_module._shared_client = _make_client(handler)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_shared_client():
    """Ensure _shared_client is None before and after every test."""
    original = wf_module._shared_client
    wf_module._shared_client = None
    yield
    wf_module._shared_client = original


@pytest.fixture(autouse=True)
def _bypass_rate_limiter(monkeypatch):
    """Skip the per-domain rate limiter so tests run instantly."""
    async def _noop(_url: str) -> None:
        pass

    monkeypatch.setattr(wf_module._DomainRateLimiter, "acquire", _noop)


# ---------------------------------------------------------------------------
# 1. URL validation rejects file:///
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reject_file_scheme():
    result = await tool.execute({"url": "file:///etc/passwd"}, {})
    assert result.startswith("Web fetch error:")


# ---------------------------------------------------------------------------
# 2. Format defaults to "markdown"
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_format_defaults_to_markdown():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text="<h1>Hello</h1><p>World</p>",
            headers={"content-type": "text/html"},
        )

    _set_shared_client(handler)
    result = await tool.execute({"url": "http://example.com"}, {})
    assert "Hello" in result
    assert "<h1>" not in result


# ---------------------------------------------------------------------------
# 3. Format "html" returns raw HTML
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_format_html_returns_raw():
    raw_html = "<h1>Hello</h1><p>World</p>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=raw_html,
            headers={"content-type": "text/html"},
        )

    _set_shared_client(handler)
    result = await tool.execute(
        {"url": "http://example.com", "format": "html"}, {}
    )
    assert result == raw_html


# ---------------------------------------------------------------------------
# 4. Timeout > 120 is clamped to 120
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_timeout_clamped_to_max():
    captured_timeout = None

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="ok", headers={"content-type": "text/plain"})

    client = _make_client(handler)

    original_get = client.get

    async def _capturing_get(url, **kwargs):
        nonlocal captured_timeout
        captured_timeout = kwargs.get("timeout")
        return await original_get(url, **kwargs)

    client.get = _capturing_get
    wf_module._shared_client = client

    await tool.execute({"url": "http://example.com", "timeout": 200}, {})

    assert captured_timeout is not None
    assert captured_timeout.connect == 120.0


# ---------------------------------------------------------------------------
# 5. Response > 5 MB triggers size limit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_response_too_large():
    five_mb_plus = 5 * 1024 * 1024 + 1

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"x" * five_mb_plus,
            headers={"content-type": "text/html"},
        )

    _set_shared_client(handler)
    result = await tool.execute({"url": "http://example.com"}, {})
    assert "response too large" in result


# ---------------------------------------------------------------------------
# 6. Image response returns metadata string
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_image_returns_metadata():
    png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 1024

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=png_data,
            headers={"content-type": "image/png"},
        )

    _set_shared_client(handler)
    result = await tool.execute({"url": "http://example.com/img.png"}, {})
    assert result.startswith("Fetched image:")


# ---------------------------------------------------------------------------
# 7. SVG is NOT treated as binary image
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_svg_returns_text():
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><circle r="10"/></svg>'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=svg,
            headers={"content-type": "image/svg+xml"},
        )

    _set_shared_client(handler)
    result = await tool.execute({"url": "http://example.com/icon.svg"}, {})
    assert "Fetched image:" not in result
    assert "<svg" in result


# ---------------------------------------------------------------------------
# 8. Cloudflare 403 triggers retry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cloudflare_retry():
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(
                403,
                text="challenge",
                headers={"cf-mitigated": "challenge"},
            )
        return httpx.Response(
            200,
            text="<p>Success after retry</p>",
            headers={"content-type": "text/html"},
        )

    _set_shared_client(handler)
    result = await tool.execute({"url": "http://example.com"}, {})
    assert call_count == 2
    assert "Success after retry" in result


# ---------------------------------------------------------------------------
# 9. Non-HTML with format=markdown returns raw text
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_plain_text_with_markdown_format():
    plain = "Just some plain text, nothing fancy."

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=plain,
            headers={"content-type": "text/plain"},
        )

    _set_shared_client(handler)
    result = await tool.execute(
        {"url": "http://example.com/readme.txt", "format": "markdown"}, {}
    )
    assert result == plain


# ---------------------------------------------------------------------------
# 10. Tool auto-discovery by runner works
# ---------------------------------------------------------------------------

def test_tool_discovery():
    import engine.runner as runner_mod
    runner_mod._custom_tools_cache = None

    tools = _discover_custom_tools()
    tool_names = [t.name for t in tools]
    assert "web_fetch" in tool_names


# ---------------------------------------------------------------------------
# 11. Timeout on slow server
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_timeout_exception():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("slow")

    _set_shared_client(handler)
    result = await tool.execute({"url": "http://example.com"}, {})
    assert "timed out" in result


# ---------------------------------------------------------------------------
# 12. Network error returns error string
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_network_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    _set_shared_client(handler)
    result = await tool.execute({"url": "http://example.com"}, {})
    assert "network request failed" in result
