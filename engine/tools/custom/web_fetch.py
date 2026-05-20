"""Web fetch tool — retrieve URL content and convert to markdown/text/html.

Fetches a given URL, converts the response body to the requested format,
and returns the result as a string. Handles HTML-to-Markdown conversion,
plain-text extraction, image detection, Cloudflare challenges, and response
size limits.
"""

import asyncio
from html.parser import HTMLParser
from typing import List
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from engine.logging import get_logger
from engine.safety import ResultTruncator
from engine.tools.base import Tool

# Module-level shared httpx.AsyncClient for connection pooling.

_shared_client: httpx.AsyncClient | None = None
_client_lock: asyncio.Lock = asyncio.Lock()


async def _get_client() -> httpx.AsyncClient:
    """Lazy-init a shared httpx.AsyncClient with connection pooling."""
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        async with _client_lock:
            if _shared_client is None or _shared_client.is_closed:
                _shared_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(30.0),
                    follow_redirects=True,
                )
    return _shared_client


# Per-domain rate limiter — 1s minimum interval between requests to the same domain.
# Lock is held only during timestamp lookup/update; sleep happens outside the lock.

class _DomainRateLimiter:
    _domain_timestamps: dict[str, float] = {}
    _lock: asyncio.Lock = asyncio.Lock()
    MIN_INTERVAL: float = 1.0

    @classmethod
    async def acquire(cls, url: str) -> None:
        domain = urlparse(url).netloc
        async with cls._lock:
            now = asyncio.get_event_loop().time()
            last_time = cls._domain_timestamps.get(domain, 0.0)
            wait_time = max(0, last_time + cls.MIN_INTERVAL - now)
            cls._domain_timestamps[domain] = now + wait_time
        if wait_time > 0:
            await asyncio.sleep(wait_time)


# HTML → plain-text helper

_SKIP_TAGS = frozenset({"script", "style", "noscript", "iframe", "object", "embed"})


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip = False
        self._parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[tuple]) -> None:
        if tag in _SKIP_TAGS:
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts).strip()


def _extract_text_from_html(html: str) -> str:
    extractor = _TextExtractor()
    extractor.feed(html)
    return extractor.get_text()


def _convert_html_to_markdown(html: str) -> str:
    return md(html, heading_style="ATX", bullets="-")


# ---------------------------------------------------------------------------
# HTML noise filter — remove navigation, ads, cookie banners, etc.
# ---------------------------------------------------------------------------

_NOISE_TAGS = frozenset({"nav", "footer", "header", "aside"})

_NOISE_CLASS_PATTERNS = (
    "ad", "ads", "advert", "advertisement", "sponsor",
    "cookie", "gdpr", "consent", "banner", "popup", "notice",
    "share", "social", "twitter", "facebook", "linkedin",
    "comment", "disqus", "respond",
)

_NOISE_ID_PATTERNS = (
    "ad", "ads", "cookie", "gdpr", "consent", "banner", "popup",
    "share", "social", "comment", "disqus",
)


def _has_noise_class(tag) -> bool:
    """Check if a tag's class list contains any noise-related patterns."""
    classes = tag.get("class", [])
    if isinstance(classes, str):
        classes = [classes]
    combined = " ".join(classes).lower()
    return any(pat in combined for pat in _NOISE_CLASS_PATTERNS)


def _has_noise_id(tag) -> bool:
    """Check if a tag's id contains any noise-related patterns."""
    tag_id = (tag.get("id") or "").lower()
    return any(pat in tag_id for pat in _NOISE_ID_PATTERNS)


def _filter_html_noise(html: str) -> str:
    """Remove noise elements from HTML (nav, ads, cookie banners, etc.).

    Uses BeautifulSoup to parse and decompose noise elements before
    format conversion. On any parsing failure, returns the original
    HTML unchanged as a safe fallback.
    """
    try:
        soup = BeautifulSoup(html, "html.parser")

        # 1. Remove structural noise tags entirely
        for tag in soup.find_all(_NOISE_TAGS):
            tag.decompose()

        # 2. Remove elements with noise-related class or id patterns
        for tag in soup.find_all(True):
            if _has_noise_class(tag) or _has_noise_id(tag):
                tag.decompose()

        # 3. Remove script/style/noscript/iframe (supplementary to markdownify)
        for tag in soup.find_all(["script", "style", "noscript", "iframe"]):
            tag.decompose()

        return str(soup)
    except Exception:
        return html


# ---------------------------------------------------------------------------
# Content pagination — offset/limit based, line-numbered output
# ---------------------------------------------------------------------------


def _paginate_content(
    content: str, offset: int, limit: int
) -> tuple[str, int]:
    """Paginate content by lines with line-number prefixes.

    Args:
        content: The full text content to paginate.
        offset: 1-indexed line number to start from.
        limit: Maximum number of lines to return.

    Returns:
        Tuple of (paginated_output, total_lines).
    """
    lines = content.splitlines()
    total_lines = len(lines)

    if offset > total_lines:
        return (
            f"[Offset {offset} exceeds total lines ({total_lines}). "
            f"Content ends at line {total_lines}.]",
            total_lines,
        )

    end = min(offset + limit - 1, total_lines)
    page_lines = []
    for i in range(offset - 1, end):
        page_lines.append(f"{i + 1}: {lines[i]}")

    output = "\n".join(page_lines)

    shown_end = offset + len(page_lines) - 1
    if total_lines > shown_end:
        footer = (
            f"\n\n[Showing lines {offset}-{shown_end} of {total_lines}. "
            f"Use offset={shown_end + 1} to continue.]"
        )
        output += footer

    return output, total_lines


# Format-dependent Accept headers

_ACCEPT_HEADERS = {
    "markdown": "text/markdown;q=1.0, text/x-markdown;q=0.9, text/plain;q=0.8, text/html;q=0.7, */*;q=0.1",
    "text": "text/plain;q=1.0, text/markdown;q=0.9, text/html;q=0.8, */*;q=0.1",
    "html": "text/html;q=1.0, application/xhtml+xml;q=0.9, text/plain;q=0.8, text/markdown;q=0.7, */*;q=0.1",
}

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
)
_HONEST_UA = "engine/0.1"


class WebFetchTool(Tool):
    """Fetch a URL and return its content in the configured format."""

    DEFAULT_FORMAT: str = "markdown"

    name = "web_fetch"
    short_description = "Fetch and convert web pages to markdown or text"
    description = (
        "Fetches content from a specified URL and returns it in the requested format. "
        "Use this tool when you need to retrieve and analyze web page content, read "
        "documentation, extract text from a webpage, or inspect the raw HTML of a URL.\n\n"
        "Usage notes:\n"
        "- The URL must be a fully-formed valid URL starting with http:// or https://.\n"
        "- Content format is determined by developer configuration (default: Markdown).\n"
        "- This tool is read-only and does not modify any files.\n"
        "- HTML noise (navigation, ads, cookie banners, social sharing) is automatically "
        "filtered out to return cleaner content.\n"
        "- For large pages, use offset and limit to read in segments. A footer will "
        "indicate how many lines remain and what offset to use next.\n"
        "- Response size limit: 5 MB.\n"
        "- Default timeout: 30 seconds (configurable, max 120 seconds).\n"
        "- If the URL points to an image (e.g. PNG, JPEG, GIF, WebP), returns a description "
        "with the image metadata instead of binary content.\n"
        "- SVG images are returned as text content, not treated as binary.\n"
        "- IMPORTANT: if another tool is present that offers better web fetching capabilities, "
        "is more targeted to the task, or has fewer restrictions, prefer using that tool instead."
    )

    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": (
                    "The fully-formed URL to fetch content from. Must start with "
                    "http:// or https://. Examples: \"https://example.com\", "
                    "\"https://docs.python.org/3/library/asyncio.html\""
                ),
            },
            "timeout": {
                "type": "number",
                "description": (
                    "Optional timeout in seconds for the HTTP request. "
                    "Default: 30 seconds. Maximum allowed: 120 seconds. "
                    "Increase this for slow or large pages."
                ),
            },
            "offset": {
                "type": "integer",
                "description": (
                    "1-indexed line number to start reading from. "
                    "Default: 1 (start from the beginning)."
                ),
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Maximum number of lines to read. Default: 500. "
                    "Use with offset to read large pages in segments."
                ),
            },
        },
        "required": ["url"],
    }

    _MAX_RESPONSE_SIZE: int = 5 * 1024 * 1024
    _DEFAULT_TIMEOUT: int = 30
    _MAX_TIMEOUT: int = 120
    _MAX_CONTENT_LENGTH: int = 15000
    _MAX_RETRIES: int = 1
    _RETRY_DELAY: float = 2.0
    DEFAULT_PAGE_LIMIT: int = 500

    async def execute(self, arguments: dict, context: dict) -> str:
        url = arguments.get("url", "")
        if not url or not isinstance(url, str) or not url.strip():
            return "Web fetch error: empty URL provided"
        url = url.strip()

        if not url.startswith(("http://", "https://")):
            return "Web fetch error: URL must start with http:// or https://"

        fmt = self.DEFAULT_FORMAT
        if fmt not in ("markdown", "text", "html"):
            fmt = "markdown"

        raw_timeout = arguments.get("timeout", self._DEFAULT_TIMEOUT)
        try:
            timeout_val = int(raw_timeout) if raw_timeout is not None else self._DEFAULT_TIMEOUT
        except (ValueError, TypeError):
            timeout_val = self._DEFAULT_TIMEOUT
        timeout_seconds = min(timeout_val, self._MAX_TIMEOUT)
        timeout = httpx.Timeout(timeout_seconds)

        headers = {
            "User-Agent": _BROWSER_UA,
            "Accept": _ACCEPT_HEADERS.get(fmt, _ACCEPT_HEADERS["markdown"]),
            "Accept-Language": "en-US,en;q=0.9",
        }

        for attempt in range(self._MAX_RETRIES + 1):
            await _DomainRateLimiter.acquire(url)

            try:
                client = await _get_client()
                response = await client.get(url, headers=headers, timeout=timeout)
            except httpx.TimeoutException:
                if attempt < self._MAX_RETRIES:
                    get_logger().warning(
                        "WebFetch",
                        "Retry {}/{} after timeout | url={:.80s}".format(
                            attempt + 1, self._MAX_RETRIES, url
                        ),
                        event_type="tool_retry",
                        data={"attempt": attempt + 1, "error": "timeout", "url": url},
                    )
                    await asyncio.sleep(self._RETRY_DELAY)
                    continue
                return "Web fetch error: request timed out"
            except httpx.RequestError:
                if attempt < self._MAX_RETRIES:
                    get_logger().warning(
                        "WebFetch",
                        "Retry {}/{} after network error | url={:.80s}".format(
                            attempt + 1, self._MAX_RETRIES, url
                        ),
                        event_type="tool_retry",
                        data={"attempt": attempt + 1, "error": "network", "url": url},
                    )
                    await asyncio.sleep(self._RETRY_DELAY)
                    continue
                return "Web fetch error: network request failed"
            except Exception as exc:
                if attempt < self._MAX_RETRIES:
                    get_logger().warning(
                        "WebFetch",
                        "Retry {}/{} after unexpected error | url={:.80s}".format(
                            attempt + 1, self._MAX_RETRIES, url
                        ),
                        event_type="tool_retry",
                        data={"attempt": attempt + 1, "error": str(exc), "url": url},
                    )
                    await asyncio.sleep(self._RETRY_DELAY)
                    continue
                return f"Web fetch error: unexpected failure - {exc}"

            break

        # Cloudflare challenge — retry with honest UA
        if response.status_code == 403 and response.headers.get("cf-mitigated") == "challenge":
            try:
                retry_headers = {**headers, "User-Agent": _HONEST_UA}
                response = await client.get(url, headers=retry_headers, timeout=timeout)
            except httpx.TimeoutException:
                return "Web fetch error: request timed out"
            except httpx.RequestError:
                return "Web fetch error: network request failed"
            except Exception as exc:
                return f"Web fetch error: unexpected failure on retry - {exc}"

        if response.status_code >= 400:
            return f"Web fetch error: HTTP {response.status_code}"

        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > self._MAX_RESPONSE_SIZE:
            return "Web fetch error: response too large (exceeds 5MB limit)"

        if len(response.content) > self._MAX_RESPONSE_SIZE:
            return "Web fetch error: response too large (exceeds 5MB limit)"

        content_type = response.headers.get("content-type", "")
        mime = content_type.split(";")[0].strip().lower()

        if mime.startswith("image/") and mime != "image/svg+xml":
            size_kb = len(response.content) / 1024
            return f"Fetched image: {url} ({mime}, {size_kb:.1f}KB)"

        if mime and not mime.startswith("text/") and not mime.startswith("image/") and mime != "application/json":
            return f"Web fetch error: unsupported content type: {mime}"

        content = response.text
        is_html = "text/html" in content_type

        # Extract pagination parameters
        offset = max(1, arguments.get("offset", 1) or 1)
        limit = arguments.get("limit", self.DEFAULT_PAGE_LIMIT) or self.DEFAULT_PAGE_LIMIT

        # Apply noise filtering for HTML content (before format conversion)
        if is_html:
            content = _filter_html_noise(content)

        # Format conversion
        if fmt == "markdown" and is_html:
            output = _convert_html_to_markdown(content)
        elif fmt == "text" and is_html:
            output = _extract_text_from_html(content)
        else:
            output = content

        if not output or not output.strip():
            return "[Empty page] The URL returned no extractable content — the page may require JavaScript rendering."

        # Apply pagination
        output, total_lines = _paginate_content(output, offset, limit)

        # Final safety net truncation
        return ResultTruncator.truncate(output, self._MAX_CONTENT_LENGTH)
