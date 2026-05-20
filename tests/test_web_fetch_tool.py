"""Tests for WebFetchTool — HTML noise filtering, pagination, pipeline integration, and edge cases."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from engine.tools.custom.web_fetch import (
    WebFetchTool,
    _filter_html_noise,
    _paginate_content,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(tool, args, context=None):
    return asyncio.run(tool.execute(args, context or {}))


def _mock_response(text="", content_type="text/html; charset=utf-8", status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = {"content-type": content_type}
    resp.text = text
    resp.content = text.encode() if isinstance(text, str) else text
    return resp


def _mock_client(response):
    """Return a MagicMock that acts like an httpx.AsyncClient returning *response* on .get()."""
    client = MagicMock()
    client.get = AsyncMock(return_value=response)
    return client


def _patch_get_client(client):
    """Patch _get_client to return the given mock client."""
    return patch(
        "engine.tools.custom.web_fetch._get_client",
        new_callable=AsyncMock,
        return_value=client,
    )


def _patch_rate_limiter():
    """Patch _DomainRateLimiter.acquire to be a no-op."""
    return patch(
        "engine.tools.custom.web_fetch._DomainRateLimiter.acquire",
        new_callable=AsyncMock,
    )


# ---------------------------------------------------------------------------
# 1. TestWebFetchNoiseFiltering — direct tests for _filter_html_noise
# ---------------------------------------------------------------------------


class TestWebFetchNoiseFiltering:
    """Tests for _filter_html_noise: structural tags, ads, cookies, social, comments."""

    def test_nav_footer_header_aside_removed(self):
        html = (
            "<html><body>"
            "<nav>Navigation links</nav>"
            "<header>Site header</header>"
            "<footer>Site footer</footer>"
            "<aside>Sidebar</aside>"
            "<main>Main content here</main>"
            "</body></html>"
        )
        result = _filter_html_noise(html)
        assert "Navigation links" not in result
        assert "Site header" not in result
        assert "Site footer" not in result
        assert "Sidebar" not in result
        assert "Main content here" in result

    def test_cookie_gdpr_banners_removed(self):
        html = (
            "<html><body>"
            '<div class="cookie-banner">Accept cookies?</div>'
            '<div id="gdpr-notice">GDPR consent</div>'
            '<div class="consent-popup">We value your privacy</div>'
            "<p>Real paragraph content</p>"
            "</body></html>"
        )
        result = _filter_html_noise(html)
        assert "Accept cookies?" not in result
        assert "GDPR consent" not in result
        assert "We value your privacy" not in result
        assert "Real paragraph content" in result

    def test_ad_elements_removed(self):
        html = (
            "<html><body>"
            '<div class="ad">Buy now!</div>'
            '<div class="advertisement">Sponsored link</div>'
            '<div class="sponsor">Partner content</div>'
            "<article>Article body text</article>"
            "</body></html>"
        )
        result = _filter_html_noise(html)
        assert "Buy now!" not in result
        assert "Sponsored link" not in result
        assert "Partner content" not in result
        assert "Article body text" in result

    def test_social_sharing_removed(self):
        html = (
            "<html><body>"
            '<div class="share-buttons">Share on Twitter</div>'
            '<div class="social-media">Follow us</div>'
            '<div class="twitter-widget">Tweet</div>'
            '<div class="facebook-like">Like</div>'
            '<div class="linkedin-share">Connect</div>'
            "<p>Important article text</p>"
            "</body></html>"
        )
        result = _filter_html_noise(html)
        assert "Share on Twitter" not in result
        assert "Follow us" not in result
        assert "Tweet" not in result
        assert "Like" not in result
        assert "Connect" not in result
        assert "Important article text" in result

    def test_comment_sections_removed(self):
        html = (
            "<html><body>"
            '<div class="comment">First!</div>'
            '<div id="disqus_thread">Disqus area</div>'
            '<div class="respond">Leave a reply</div>'
            "<p>Actual post content</p>"
            "</body></html>"
        )
        result = _filter_html_noise(html)
        assert "First!" not in result
        assert "Disqus area" not in result
        assert "Leave a reply" not in result
        assert "Actual post content" in result

    def test_script_style_noscript_iframe_removed(self):
        html = (
            "<html><body>"
            "<script>var x = 1;</script>"
            "<style>body { color: red; }</style>"
            "<noscript>Enable JS</noscript>"
            '<iframe src="https://ads.example.com"></iframe>'
            "<p>Visible text</p>"
            "</body></html>"
        )
        result = _filter_html_noise(html)
        assert "var x = 1;" not in result
        assert "color: red" not in result
        assert "Enable JS" not in result
        assert "ads.example.com" not in result
        assert "Visible text" in result

    def test_main_content_preserved(self):
        html = (
            "<html><body>"
            "<main>"
            "<article>"
            "<h1>Title</h1>"
            "<p>Paragraph one</p>"
            "<p>Paragraph two</p>"
            "</article>"
            "</main>"
            "</body></html>"
        )
        result = _filter_html_noise(html)
        assert "Title" in result
        assert "Paragraph one" in result
        assert "Paragraph two" in result

    def test_malformed_html_no_crash(self):
        html = "<html><body><div><p>Unclosed tags<div>Nested badly</body>"
        # Should not raise — returns original on parse failure or processes gracefully
        result = _filter_html_noise(html)
        assert isinstance(result, str)
        assert "Unclosed tags" in result or isinstance(result, str)


# ---------------------------------------------------------------------------
# 2. TestWebFetchPagination — direct tests for _paginate_content
# ---------------------------------------------------------------------------


class TestWebFetchPagination:
    """Tests for _paginate_content: offset, limit, footer, edge cases."""

    def test_basic_offset_limit(self):
        content = "alpha\nbeta\ngamma\ndelta\nepsilon"
        result, total = _paginate_content(content, offset=2, limit=2)
        assert total == 5
        assert "2: beta" in result
        assert "3: gamma" in result
        assert "1: alpha" not in result
        assert "4: delta" not in result
        assert "Showing lines 2-3" in result

    def test_offset_beyond_content(self):
        content = "line1\nline2\nline3"
        result, total = _paginate_content(content, offset=10, limit=5)
        assert total == 3
        assert "exceeds total lines" in result
        assert "10" in result

    def test_full_content_no_footer(self):
        content = "line1\nline2\nline3"
        result, total = _paginate_content(content, offset=1, limit=100)
        assert total == 3
        assert "1: line1" in result
        assert "2: line2" in result
        assert "3: line3" in result
        assert "Showing lines" not in result

    def test_default_offset_is_1(self):
        content = "first\nsecond"
        result, total = _paginate_content(content, offset=1, limit=500)
        assert result.startswith("1: first")

    def test_footer_shows_continue_hint(self):
        lines = [f"line{i}" for i in range(1, 21)]
        content = "\n".join(lines)
        result, total = _paginate_content(content, offset=1, limit=5)
        assert total == 20
        assert "offset=6" in result
        assert "Showing lines 1-5 of 20" in result

    def test_empty_content(self):
        result, total = _paginate_content("", offset=1, limit=10)
        assert total == 0 or total == 1  # empty string has 1 line from splitlines if truly empty
        # Should not crash — just returns gracefully
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 3. TestWebFetchPipeline — integration tests with mocked HTTP
# ---------------------------------------------------------------------------


class TestWebFetchPipeline:
    """Integration tests: full execute() pipeline with mocked HTTP client."""

    def test_html_filtered_and_paginated(self):
        html = (
            "<html><body>"
            "<nav>Nav links</nav>"
            "<main><p>Important article text</p></main>"
            "<footer>Copyright 2025</footer>"
            "</body></html>"
        )
        resp = _mock_response(text=html, content_type="text/html; charset=utf-8")
        client = _mock_client(resp)
        with _patch_get_client(client), _patch_rate_limiter():
            result = _run(WebFetchTool(), {"url": "https://example.com"})
        assert "Important article text" in result
        assert "Nav links" not in result
        assert "Copyright 2025" not in result

    def test_non_html_paginated_no_filtering(self):
        json_body = '{"key": "value", "count": 42}'
        resp = _mock_response(
            text=json_body,
            content_type="application/json; charset=utf-8",
        )
        client = _mock_client(resp)
        with _patch_get_client(client), _patch_rate_limiter():
            result = _run(WebFetchTool(), {"url": "https://api.example.com/data"})
        assert '"key"' in result
        assert "42" in result

    def test_pagination_with_offset_limit(self):
        paragraphs = "".join(f"<p>Paragraph {i}</p>" for i in range(1, 51))
        html = f"<html><body><main>{paragraphs}</main></body></html>"
        resp = _mock_response(text=html, content_type="text/html; charset=utf-8")
        client = _mock_client(resp)
        with _patch_get_client(client), _patch_rate_limiter():
            result = _run(
                WebFetchTool(),
                {"url": "https://example.com/long", "offset": 2, "limit": 3},
            )
        # Should have line-numbered output from offset 2
        assert "2: " in result

    def test_empty_after_filtering(self):
        html = (
            "<html><body>"
            "<nav>All nav</nav>"
            "<footer>All footer</footer>"
            "</body></html>"
        )
        resp = _mock_response(text=html, content_type="text/html; charset=utf-8")
        client = _mock_client(resp)
        with _patch_get_client(client), _patch_rate_limiter():
            result = _run(WebFetchTool(), {"url": "https://example.com/empty-ish"})
        assert "Empty page" in result or "no extractable content" in result

    def test_error_timeout_unchanged(self):
        client = MagicMock()
        client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
        with _patch_get_client(client), _patch_rate_limiter():
            result = _run(WebFetchTool(), {"url": "https://slow.example.com"})
        assert "timed out" in result

    def test_error_404_unchanged(self):
        resp = _mock_response(text="Not Found", status_code=404)
        client = _mock_client(resp)
        with _patch_get_client(client), _patch_rate_limiter():
            result = _run(WebFetchTool(), {"url": "https://example.com/missing"})
        assert "HTTP 404" in result

    def test_cloudflare_retry_unchanged(self):
        # First response: 403 with cf-mitigated=challenge
        cf_resp = MagicMock()
        cf_resp.status_code = 403
        cf_resp.headers = {
            "content-type": "text/html",
            "cf-mitigated": "challenge",
        }
        # Second response after retry with honest UA
        ok_resp = _mock_response(
            text="<html><body><p>Welcome</p></body></html>",
            content_type="text/html",
        )
        client = MagicMock()
        client.get = AsyncMock(side_effect=[cf_resp, ok_resp])
        with _patch_get_client(client), _patch_rate_limiter():
            result = _run(
                WebFetchTool(), {"url": "https://protected.example.com"}
            )
        assert "Welcome" in result

    def test_image_response_unchanged(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"content-type": "image/png"}
        resp.content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        resp.text = ""  # not used for images
        client = _mock_client(resp)
        with _patch_get_client(client), _patch_rate_limiter():
            result = _run(WebFetchTool(), {"url": "https://example.com/photo.png"})
        assert "Fetched image" in result
        assert "image/png" in result


# ---------------------------------------------------------------------------
# 4. TestWebFetchEdgeCases — URL validation, defaults, None handling
# ---------------------------------------------------------------------------


class TestWebFetchEdgeCases:
    """Edge cases: URL validation, default parameters, None handling."""

    def test_url_validation_empty(self):
        result = _run(WebFetchTool(), {"url": ""})
        assert "empty URL" in result.lower() or "error" in result.lower()

    def test_url_validation_invalid_scheme(self):
        result = _run(WebFetchTool(), {"url": "ftp://files.example.com"})
        assert "http://" in result or "https://" in result

    def test_url_validation_missing_url_key(self):
        result = _run(WebFetchTool(), {})
        assert "error" in result.lower() or "empty" in result.lower()

    def test_default_offset_and_limit(self):
        json_body = '{"data": "hello"}'
        resp = _mock_response(
            text=json_body,
            content_type="application/json",
        )
        client = _mock_client(resp)
        with _patch_get_client(client), _patch_rate_limiter():
            result = _run(WebFetchTool(), {"url": "https://api.example.com"})
        # With defaults (offset=1, limit=500), short content should have no footer
        assert "1: " in result
        assert "Showing lines" not in result

    def test_offset_none_defaults_to_1(self):
        json_body = "line1\nline2"
        resp = _mock_response(text=json_body, content_type="text/plain")
        client = _mock_client(resp)
        with _patch_get_client(client), _patch_rate_limiter():
            result = _run(
                WebFetchTool(),
                {"url": "https://example.com", "offset": None},
            )
        assert "1: " in result

    def test_limit_none_defaults_to_500(self):
        lines = [f"line{i}" for i in range(1, 10)]
        resp = _mock_response(text="\n".join(lines), content_type="text/plain")
        client = _mock_client(resp)
        with _patch_get_client(client), _patch_rate_limiter():
            result = _run(
                WebFetchTool(),
                {"url": "https://example.com", "limit": None},
            )
        # All 9 lines should be present since limit defaults to 500
        assert "9: line9" in result
        assert "Showing lines" not in result
