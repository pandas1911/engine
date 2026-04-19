# Tavily Search & Extract Tools Implementation

## TL;DR

> **Quick Summary**: Implement two web retrieval tools (Search and Extract) for the engine agent framework using Tavily's REST API with a custom stdlib-based HTTP client. The client is designed for easy porting to Swift (URLSession).
> 
> **Deliverables**:
> - `TavilyClient` HTTP client class (stdlib urllib, async-safe)
> - `SearchTool` (Tool subclass) — web search via `POST /search`
> - `ExtractTool` (Tool subclass) — URL content extraction via `POST /extract`
> - Unit tests with mocked HTTP responses
> - Updated `.env.example` with `TAVILY_API_KEY`
> 
> **Estimated Effort**: Quick-Medium
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Task 1 → Task 2/3 (parallel) → Task 4 → Task 5

---

## Context

### Original Request
Design two联网检索 tools using Tavily API: Search and Extract. Must NOT use tavily SDK — implement a custom HTTP client because it needs to be ported to Swift later.

### Interview Summary
**Key Discussions**:
- HTTP Client: stdlib urllib chosen (zero dependency, maps cleanly to Swift URLSession)
- Parameter exposure: Simplified core parameters only, not all 20+ API params
- File organization: Single file `engine/tools/custom/tavily.py` (auto-discovered by framework)
- Result format: JSON string returned directly to LLM
- API Key: Independent `os.getenv("TAVILY_API_KEY")`, NOT extending Config dataclass
- Testing: Unit tests with mocked HTTP calls using pytest

**Research Findings**:
- Tool ABC pattern: `name`, `description`, `parameters` (OpenAI function calling schema), `async execute(arguments, context) -> str`
- Custom tools auto-discovered from `engine/tools/custom/*.py` — any class subclassing `Tool` is auto-instantiated
- Framework's `_execute_tool()` catches exceptions and returns error strings — tools should catch and return structured errors
- `urllib` is synchronous, but `execute()` is async → need `asyncio.to_thread()` wrapper
- Existing `max_result_length` in Config defaults to 4000 chars

### Metis Review
**Identified Gaps** (addressed):
- async/urllib mismatch → resolved: use `asyncio.to_thread()` for non-blocking
- Tool naming collision risk → resolved: prefix as `tavily_search`, `tavily_extract`
- HTTP timeout → resolved: default 30s, configurable
- Scope creep (caching, retry, streaming) → locked down: none in v1
- Result truncation → resolved: cap at reasonable length, truncate with marker

---

## Work Objectives

### Core Objective
Add two production-ready web retrieval tools to the engine framework that agents can use to search the web and extract page content, using a custom HTTP client designed for Swift portability.

### Concrete Deliverables
- `engine/tools/custom/tavily.py` — TavilyClient + SearchTool + ExtractTool
- `tests/test_tavily_tools.py` — Unit tests with mocked HTTP
- `.env.example` updated with `TAVILY_API_KEY`

### Definition of Done
- [ ] `SearchTool` and `ExtractTool` are auto-discovered and registered by `_discover_custom_tools()`
- [ ] `SearchTool.execute()` calls `POST /search` and returns JSON string
- [ ] `ExtractTool.execute()` calls `POST /extract` and returns JSON string
- [ ] All HTTP calls use stdlib urllib only (no external HTTP dependencies)
- [ ] HTTP calls are wrapped in `asyncio.to_thread()` to avoid blocking event loop
- [ ] API key read from `os.getenv("TAVILY_API_KEY")`
- [ ] Error responses are structured JSON strings, not raised exceptions
  - [ ] Unit tests mock urllib at HTTP level, all pass → Integration tests use real Tavily API, all pass

### Must Have
- TavilyClient with `_request()` method handling POST, auth header, JSON encode/decode
- SearchTool with OpenAI function calling schema for core params
- ExtractTool with OpenAI function calling schema for core params
- Proper Bearer token auth
- HTTP timeout (default 30s)
- Error handling for Tavily API error codes (400, 401, 429, 432, 433, 500)
- Result truncation for oversized responses
- `asyncio.to_thread()` wrapper for sync urllib calls
- Unit tests covering happy path + error cases

### Must NOT Have (Guardrails)
- NO tavily SDK or any new dependency in pyproject.toml
- NO modifications to `engine/config.py` (Config dataclass stays unchanged)
- NO modifications to `engine/tools/base.py` (Tool ABC stays unchanged)
- NO retry logic, exponential backoff, or caching
- NO streaming support
- NO URL validation beyond basic presence check
- NO result filtering or ranking on the client side
- NO AI slop: excessive comments, over-abstraction, unnecessary error classes

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (pytest + pytest-asyncio in pyproject.toml)
- **Automated tests**: YES (Tests-after) — **Integration tests with real Tavily API**
- **Framework**: pytest
- **API Key**: Real key in `.env` (already gitignored), tests read via `os.getenv("TAVILY_API_KEY")`
- **No mocking**: All tests make actual HTTP calls to Tavily API

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Library/Module**: Use Bash (python REPL / pytest) — Import, call functions, verify output

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately - foundation):
└── Task 1: TavilyClient HTTP client [deep]

Wave 2 (After Wave 1 - tools + config, PARALLEL):
├── Task 2: SearchTool (depends: 1) [quick]
├── Task 3: ExtractTool (depends: 1) [quick]
└── Task 4: Update .env.example (depends: none, but groups with 2,3) [quick]

Wave 3 (After Wave 2 - tests):
└── Task 5: Unit tests with mocked HTTP (depends: 1, 2, 3) [unspecified-high]

Wave FINAL (After ALL tasks — 4 parallel reviews):
├── F1: Plan compliance audit (oracle)
├── F2: Code quality review (unspecified-high)
├── F3: Real manual QA (unspecified-high)
└── F4: Scope fidelity check (deep)
→ Present results → Get explicit user okay

Critical Path: Task 1 → Task 2/3 → Task 5 → F1-F4
Parallel Speedup: Tasks 2, 3, 4 can run in parallel
Max Concurrent: 3 (Wave 2)
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|-----------|--------|------|
| 1    | -         | 2, 3, 5 | 1   |
| 2    | 1         | 5       | 2   |
| 3    | 1         | 5       | 2   |
| 4    | -         | 5       | 2   |
| 5    | 1, 2, 3, 4| F1-F4   | 3   |

### Agent Dispatch Summary

- **Wave 1**: 1 task — T1 → `deep`
- **Wave 2**: 3 tasks — T2 → `quick`, T3 → `quick`, T4 → `quick`
- **Wave 3**: 1 task — T5 → `unspecified-high`
- **FINAL**: 4 tasks — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [ ] 1. TavilyClient HTTP Client Implementation

  **What to do**:
  - Create `engine/tools/custom/tavily.py` with a `TavilyClient` class
  - The class should have:
    - `__init__(self, api_key: str, base_url: str = "https://api.tavily.com", timeout: int = 30)`
    - `async def _request(self, endpoint: str, payload: dict) -> dict` — wraps sync HTTP in `asyncio.to_thread()`
    - Internal sync method `_post(url, headers, data, timeout)` using `urllib.request.Request` + `urlopen`
  - Auth: set `Authorization: Bearer {api_key}` header
  - JSON encoding: use `json.dumps()` for request body, `json.loads()` for response
  - Error handling: catch `urllib.error.HTTPError`, read response body for error detail, return structured dict `{"error": "...", "status_code": N}`
  - Catch `urllib.error.URLError` for network errors
  - Result truncation: if response JSON > 8000 chars when serialized, truncate `results` items' content fields
  - Do NOT add any external dependencies — use only stdlib (`urllib.request`, `urllib.error`, `json`, `asyncio`)
  - Keep the client minimal and Swift-portable: one file, simple POST, no fancy abstractions

  **Must NOT do**:
  - Add dependencies to pyproject.toml
  - Use httpx, requests, aiohttp, or any external HTTP library
  - Implement retry logic, caching, or streaming
  - Over-abstract with multiple layers (one client class is enough)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Requires understanding async/sync bridge, HTTP error handling, and Swift portability constraints
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (foundation for Tasks 2, 3)
  - **Parallel Group**: Wave 1 (alone)
  - **Blocks**: Tasks 2, 3, 5
  - **Blocked By**: None

  **References**:

  **Pattern References** (existing code to follow):
  - `engine/tools/base.py:10-31` — Tool ABC interface (`name`, `description`, `parameters`, `async execute()`)
  - `engine/tools/custom/__init__.py` — Empty init file, confirms this is the custom tools directory
  - `engine/llm_provider.py:79-93` — Pattern for initializing a provider with config/API key

  **API/Type References**:
  - Tavily API base URL: `https://api.tavily.com`
  - Auth header format: `Authorization: Bearer tvly-YOUR_API_KEY`
  - Both endpoints use `POST` with `Content-Type: application/json`
  - Error response shape: `{"detail": {"error": "message"}}`

  **External References**:
  - Python stdlib `urllib.request`: https://docs.python.org/3/library/urllib.request.html
  - Python `asyncio.to_thread()`: https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread

  **WHY Each Reference Matters**:
  - `engine/tools/base.py` — The executor must understand the Tool ABC to subclass it correctly in Tasks 2/3
  - `engine/llm_provider.py` — Shows how the project structures provider initialization with config values
  - Tavily API docs — The executor needs exact endpoint URLs, auth format, and error shapes

  **Acceptance Criteria**:

  - [ ] `engine/tools/custom/tavily.py` file created with `TavilyClient` class
  - [ ] `TavilyClient._request()` uses `asyncio.to_thread()` for non-blocking HTTP
  - [ ] Auth header correctly set as `Bearer` token
  - [ ] HTTP errors (4xx, 5xx) caught and returned as structured dict, not raised
  - [ ] Network errors caught and returned as structured dict
  - [ ] No imports from external HTTP libraries (httpx, requests, aiohttp)

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Client initializes correctly with API key
    Tool: Bash (python -c)
    Preconditions: File engine/tools/custom/tavily.py exists
    Steps:
      1. Run: python -c "from engine.tools.custom.tavily import TavilyClient; c = TavilyClient('test-key'); print(c._base_url, c._timeout)"
    Expected Result: Output contains "https://api.tavily.com" and "30"
    Failure Indicators: ImportError, AttributeError, or wrong values
    Evidence: .sisyphus/evidence/task-1-client-init.txt

  Scenario: HTTP error is caught, not raised
    Tool: Bash (python -c)
    Preconditions: TavilyClient exists
    Steps:
      1. Write a quick inline test that mocks urlopen to raise HTTPError with status 401
      2. Call _post() and verify it returns a dict with "error" and "status_code" keys
    Expected Result: Returns {"error": "...", "status_code": 401} — no exception raised
    Failure Indicators: Exception raised or missing error/status_code keys
    Evidence: .sisyphus/evidence/task-1-error-handling.txt
  ```

  **Commit**: YES (groups with Task 2, 3)
  - Message: `feat(tools): add tavily client and search/extract tools`
  - Files: `engine/tools/custom/tavily.py`
  - Pre-commit: `python -c "from engine.tools.custom.tavily import TavilyClient; print('OK')"`

- [ ] 2. SearchTool Implementation

  **What to do**:
  - In `engine/tools/custom/tavily.py`, add `SearchTool` class extending `Tool` from `engine.tools.base`
  - Class attributes:
    - `name = "tavily_search"`
    - `description` — See below
    - `parameters` — See the exact JSON schema below
  - **Tool description** (for `description` attribute):
    ```
    Search the web for information. Given a query, returns up to 20 search results — each with title, URL, and a content summary. Supports filtering by domain, topic (general/news/finance), and time range. Use this tool to discover information or find sources. If you already have a specific URL and want its full content, use tavily_extract instead.
    ```
  - **Exact `parameters` JSON schema** (copy this into the code — every `"description"` field is what the LLM sees and must be precise):
    ```json
    {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "The search query. Write a specific, descriptive query for best results — e.g. 'Python asyncio tutorial for beginners' is better than 'Python'."
        },
        "search_depth": {
          "type": "string",
          "enum": ["basic", "advanced", "fast", "ultra-fast"],
          "description": "Search depth controls result quality vs speed. 'basic': balanced quality and speed, returns one summary per URL (1 credit). 'advanced': highest relevance, returns multiple content snippets per URL, slower (2 credits). 'fast': lower latency, multiple snippets per URL (1 credit). 'ultra-fast': minimum latency, one summary per URL (1 credit). Default is 'basic'."
        },
        "topic": {
          "type": "string",
          "enum": ["general", "news", "finance"],
          "description": "Search category. Use 'general' for broad web search. Use 'news' for real-time news about politics, sports, current events. Use 'finance' for financial and market data."
        },
        "max_results": {
          "type": "integer",
          "description": "Maximum number of search results to return. Range 1-20. Default is 5. Use fewer results (1-3) for focused answers; use more (10-20) for broad research."
        },
        "include_domains": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Only return results from these domains. Useful when you know the exact source you want. Example: ['wikipedia.org', 'github.com']. Maximum 300 domains."
        },
        "exclude_domains": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Exclude results from these domains. Useful to filter out low-quality or paywalled sources. Example: ['pinterest.com', 'facebook.com']. Maximum 150 domains."
        },
        "time_range": {
          "type": "string",
          "enum": ["day", "week", "month", "year"],
          "description": "Filter results by recency. Only return sources published or updated within this time range from now. Use 'day' or 'week' for breaking news, 'month' for recent trends, 'year' for established content."
        }
      },
      "required": ["query"]
    }
    ```
  - `async def execute(self, arguments, context) -> str`:
    - Read API key from `os.getenv("TAVILY_API_KEY")`
    - If missing, return JSON error string: `{"error": "TAVILY_API_KEY not configured"}`
    - Build payload from arguments (only include non-None optional params)
    - Call `TavilyClient.search(payload)` (or client._request("search", payload))
    - Return `json.dumps(result)` (the full API response as JSON string)

  **Must NOT do**:
  - Add new dependencies
  - Modify Tool ABC or any core files
  - Add retry logic or caching
  - Expose parameters beyond the listed core set

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Straightforward Tool subclass following established pattern
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 3, Task 4)
  - **Blocks**: Task 5
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `engine/tools/base.py:10-31` — Tool ABC: must subclass and implement `execute()`
  - `engine/tools/custom/` — Target directory for auto-discovered tools
  - `engine/__init__.py:27-57` — `_discover_custom_tools()` logic: scans `*.py`, finds Tool subclasses, instantiates them
  - `engine/subagent/spawn.py` — Example of a Tool subclass with `name`, `description`, `parameters`, `async execute()`

  **API/Type References**:
  - Tavily Search endpoint: `POST https://api.tavily.com/search`
  - Required body field: `query` (string)
  - Optional: `search_depth`, `topic`, `max_results`, `include_domains`, `exclude_domains`, `time_range`
  - Success response: `{"query": "...", "answer": "...", "results": [...], "images": [...], "response_time": ...}`
  - Error response: `{"detail": {"error": "..."}}`

  **External References**:
  - OpenAI function calling schema format: `{"type": "object", "properties": {...}, "required": [...]}`

  **WHY Each Reference Matters**:
  - `engine/subagent/spawn.py` — The ONLY existing Tool subclass in the codebase. The executor MUST follow this pattern exactly.
  - `engine/__init__.py:27-57` — Shows auto-discovery mechanism. The class must subclass `Tool` with a no-arg `__init__` or it won't be instantiated.
  - OpenAI function calling format — The `parameters` attribute MUST follow this exact schema structure for the LLM to understand it.

  **Acceptance Criteria**:

  - [ ] `SearchTool` class exists in `engine/tools/custom/tavily.py`
  - [ ] `SearchTool.name == "tavily_search"`
  - [ ] `SearchTool.parameters` is a valid OpenAI function calling schema
  - [ ] `SearchTool.execute()` reads API key from `os.getenv("TAVILY_API_KEY")`
  - [ ] Missing API key returns `{"error": "TAVILY_API_KEY not configured"}`
  - [ ] Returns `json.dumps()` of Tavily API response

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: SearchTool is auto-discoverable
    Tool: Bash (python -c)
    Preconditions: engine/tools/custom/tavily.py exists with SearchTool
    Steps:
      1. Run: python -c "from engine import _discover_custom_tools; tools = _discover_custom_tools(); names = [t.name for t in tools]; print(names); assert 'tavily_search' in names"
    Expected Result: "tavily_search" appears in the printed list
    Failure Indicators: AssertionError or ImportError
    Evidence: .sisyphus/evidence/task-2-discovery.txt

  Scenario: SearchTool returns error when API key missing
    Tool: Bash (python -c)
    Preconditions: TAVILY_API_KEY not set in environment
    Steps:
      1. Run: python -c "import os; os.environ.pop('TAVILY_API_KEY', None); import asyncio; from engine.tools.custom.tavily import SearchTool; t = SearchTool(); result = asyncio.run(t.execute({'query': 'test'}, {})); print(result); assert 'error' in result"
    Expected Result: JSON string containing "TAVILY_API_KEY not configured"
    Failure Indicators: No error message, or exception raised
    Evidence: .sisyphus/evidence/task-2-missing-key.txt
  ```

  **Commit**: YES (groups with Task 1, 3)
  - Message: `feat(tools): add tavily client and search/extract tools`
  - Files: `engine/tools/custom/tavily.py`

- [ ] 3. ExtractTool Implementation

  **What to do**:
  - In `engine/tools/custom/tavily.py`, add `ExtractTool` class extending `Tool`
  - Class attributes:
    - `name = "tavily_extract"`
    - `description` — See below
    - `parameters` — See the exact JSON schema below
  - **Tool description** (for `description` attribute):
    ```
    Extract the full text content from web page URLs. Given one or more URLs, fetches and returns the page content in markdown format. Supports extracting tables and embedded content with advanced depth. Use this tool when you already have specific URLs and need to read their actual content. If you don't have URLs yet and need to find information first, use tavily_search instead.
    ```
  - **Exact `parameters` JSON schema** (copy this into the code — every `"description"` field is what the LLM sees):
    ```json
    {
      "type": "object",
      "properties": {
        "urls": {
          "oneOf": [
            {"type": "string"},
            {"type": "array", "items": {"type": "string"}}
          ],
          "description": "The URL or list of URLs to extract content from. Pass a single URL string for one page, or an array of URLs for batch extraction. Maximum 20 URLs per request. Example: 'https://en.wikipedia.org/wiki/Python_(programming_language)'"
        },
        "query": {
          "type": "string",
          "description": "A short description of what information you are looking for from these pages. When provided, the extracted content is reranked by relevance to this query, returning the most relevant portions first. Leave empty to get the full page content."
        },
        "extract_depth": {
          "type": "string",
          "enum": ["basic", "advanced"],
          "description": "Extraction depth. 'basic': standard extraction, faster (1 credit per 5 URLs). 'advanced': retrieves more data including tables and embedded content, higher success rate but slower (2 credits per 5 URLs). Default is 'basic'."
        },
        "format": {
          "type": "string",
          "enum": ["markdown", "text"],
          "description": "Output format of the extracted content. 'markdown' returns clean markdown (recommended for most cases). 'text' returns plain text but may increase latency. Default is 'markdown'."
        }
      },
      "required": ["urls"]
    }
    ```
  - `async def execute(self, arguments, context) -> str`:
    - Same API key pattern as SearchTool
    - Build payload, normalize `urls` to always be a list
    - Call `TavilyClient.extract(payload)`
    - Return `json.dumps(result)`

  **Must NOT do**:
  - Add new dependencies
  - Modify any core files
  - Validate URLs beyond basic presence check

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Follows same pattern as SearchTool, straightforward
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 2, Task 4)
  - **Blocks**: Task 5
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - Same as Task 2 — `engine/tools/base.py`, `engine/subagent/spawn.py`, `engine/__init__.py:27-57`

  **API/Type References**:
  - Tavily Extract endpoint: `POST https://api.tavily.com/extract`
  - Required body field: `urls` (string or array of strings)
  - Optional: `query`, `extract_depth`, `format`
  - Success response: `{"results": [...], "failed_results": [...], "response_time": ...}`
  - Error response: `{"detail": {"error": "..."}}`

  **WHY Each Reference Matters**:
  - Same pattern references as Task 2 — ensures consistency between both tools
  - Extract API `urls` accepts both string and array — the executor must handle both

  **Acceptance Criteria**:

  - [ ] `ExtractTool` class exists in `engine/tools/custom/tavily.py`
  - [ ] `ExtractTool.name == "tavily_extract"`
  - [ ] `ExtractTool.parameters` is a valid OpenAI function calling schema
  - [ ] `urls` parameter accepts both string and array in schema
  - [ ] Missing API key returns structured error JSON

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: ExtractTool is auto-discoverable
    Tool: Bash (python -c)
    Preconditions: engine/tools/custom/tavily.py exists with ExtractTool
    Steps:
      1. Run: python -c "from engine import _discover_custom_tools; tools = _discover_custom_tools(); names = [t.name for t in tools]; print(names); assert 'tavily_extract' in names"
    Expected Result: "tavily_extract" appears in the printed list
    Failure Indicators: AssertionError or ImportError
    Evidence: .sisyphus/evidence/task-3-discovery.txt

  Scenario: ExtractTool normalizes single URL string to list
    Tool: Bash (python -c)
    Preconditions: ExtractTool exists
    Steps:
      1. Run: python -c "import asyncio; from unittest.mock import patch; from engine.tools.custom.tavily import ExtractTool; t = ExtractTool(); payload = {}; # verify that passing urls='https://example.com' gets normalized to ['https://example.com'] before API call"
    Expected Result: URLs string is wrapped in a list for the API call
    Failure Indicators: String passed directly without normalization
    Evidence: .sisyphus/evidence/task-3-url-normalize.txt
  ```

  **Commit**: YES (groups with Task 1, 2)
  - Message: `feat(tools): add tavily client and search/extract tools`
  - Files: `engine/tools/custom/tavily.py`

- [ ] 4. Update Configuration Files (.env.example and .env)

  **What to do**:
  - Add to `.env.example` (placeholder):
    ```
    # Tavily Web Search Configuration
    TAVILY_API_KEY=your_tavily_api_key_here
    ```
  - Add to `.env` (real key, this file is gitignored):
    ```
    TAVILY_API_KEY=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
    ```
  - Place entries after the existing LLM configuration section
  - `.env.example` gets a comment section header; `.env` gets the actual key

  **Must NOT do**:
  - Modify any Python source files
  - Put real API key in `.env.example`
  - Put real API key in any Python source file or test file

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Two files, simple additions
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 2, Task 3)
  - **Blocks**: Task 5 (tests need the key in .env)
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `.env.example:1-15` — Existing format: section headers with `# ---` comments, KEY=placeholder pattern
  - `.gitignore:2` — `.env` is gitignored, safe for real keys

  **WHY Each Reference Matters**:
  - Must match existing .env.example formatting style
  - .gitignore confirmation ensures real key won't be committed

  **Acceptance Criteria**:

  - [ ] `.env.example` contains `TAVILY_API_KEY=your_tavily_api_key_here` with comment header
  - [ ] `.env` contains real Tavily API key (not placeholder)
  - [ ] `.env.example` does NOT contain real key (only placeholder)
  - [ ] `os.getenv("TAVILY_API_KEY")` returns the real key after loading .env

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Real API key available in environment
    Tool: Bash (python -c)
    Preconditions: .env file updated with real key
    Steps:
      1. Run: python -c "from dotenv import load_dotenv; load_dotenv(); import os; key = os.getenv('TAVILY_API_KEY'); print('Key length:', len(key) if key else 0); assert key and key.startswith('tvly-'), 'Invalid key format'"
    Expected Result: Prints "Key length: 52" and no assertion error
    Failure Indicators: Key is None or doesn't start with "tvly-"
    Evidence: .sisyphus/evidence/task-4-env-key.txt

  Scenario: .env.example has no real key
    Tool: Bash (grep)
    Preconditions: .env.example updated
    Steps:
      1. Run: grep "TAVILY_API_KEY=tvly-" .env.example
    Expected Result: No match (placeholder only)
    Failure Indicators: Real key pattern found
    Evidence: .sisyphus/evidence/task-4-no-real-key.txt
  ```

  **Commit**: YES (groups with Task 1, 2, 3)
  - Message: `feat(tools): add tavily client and search/extract tools`
  - Files: `.env.example` (NOT .env — it's gitignored)

- [ ] 5. Integration Tests with Real Tavily API

  **What to do**:
  - Create `tests/test_tavily_tools.py`
  - **Use real Tavily API calls** — API key is in `.env`, loaded via `dotenv`
  - Add `@pytest.fixture(autouse=True)` to load `.env` and set `TAVILY_API_KEY` before tests
  - Test cases for TavilyClient:
    - Successful POST request to `/search` returns parsed JSON with `results`
    - Successful POST request to `/extract` returns parsed JSON with `results`
    - HTTP error (invalid key) returns error dict with `status_code`
  - Test cases for SearchTool:
    - Happy path: search "Python programming" returns JSON string with non-empty results
    - Results contain expected fields: `query`, `results` (each has `title`, `url`, `content`)
    - With `topic: "news"` returns news-style results
  - Test cases for ExtractTool:
    - Happy path: extract `https://en.wikipedia.org/wiki/Python_(programming_language)` returns content
    - Results contain `raw_content` field with non-empty string
    - Multiple URLs: extract a list of URLs returns results for each
  - All tests must be async (`pytest-asyncio`)
  - Mark real-API tests with `@pytest.mark.asyncio`
  - Add a skip guard: if `TAVILY_API_KEY` not in env, skip with reason

  **Must NOT do**:
  - Mock urllib or HTTP calls — all tests hit the real Tavily API
  - Hardcode the API key in the test file
  - Import tavily SDK or any new test dependency
  - Make excessive API calls (keep test count reasonable, ~8-10 tests)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires async test writing, real API integration, and careful test isolation
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (sequential, depends on all prior)
  - **Blocks**: F1-F4
  - **Blocked By**: Tasks 1, 2, 3, 4

  **References**:

  **Pattern References**:
  - `tests/test_easy_task.py` — Existing test file showing project's test style
  - `tests/test_multilayer_subagent.py` — Another test showing async patterns

  **API/Type References**:
  - Real API key in `.env`: `TAVILY_API_KEY=tvly-dev-...`
  - Tavily Search response: `{"query": "...", "results": [{"title": "...", "url": "...", "content": "...", "score": 0.81}], "response_time": 1.5}`
  - Tavily Extract response: `{"results": [{"url": "...", "raw_content": "..."}], "failed_results": [], "response_time": 0.5}`

  **Test References**:
  - `pytest-asyncio` for async test functions
  - `python-dotenv` for loading `.env` in test setup

  **WHY Each Reference Matters**:
  - `tests/test_easy_task.py` — Shows the project's existing test patterns and conventions
  - Real API responses — Tests verify actual data shapes from Tavily
  - `.env` loading — Must load the real key before any test runs

  **Acceptance Criteria**:

  - [ ] `tests/test_tavily_tools.py` created
  - [ ] `pytest tests/test_tavily_tools.py -v` → all tests pass (with real API key in .env)
  - [ ] Minimum 8 test cases covering client, search, extract
  - [ ] All tests make real HTTP calls to Tavily API (no mocking)
  - [ ] API key read from environment, never hardcoded

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: All tests pass with real API
    Tool: Bash (pytest)
    Preconditions: .env has real TAVILY_API_KEY, all implementation files exist
    Steps:
      1. Run: pytest tests/test_tavily_tools.py -v
    Expected Result: All tests pass, 0 failures, output shows real test names with PASSED
    Failure Indicators: Any test shows FAILED or ERROR
    Evidence: .sisyphus/evidence/task-5-tests-pass.txt

  Scenario: Tests use real API, not mocks
    Tool: Bash (grep)
    Preconditions: test file exists
    Steps:
      1. Run: grep -c "mock\|Mock\|patch" tests/test_tavily_tools.py
    Expected Result: Returns 0 (no mocking used)
    Failure Indicators: Count > 0
    Evidence: .sisyphus/evidence/task-5-no-mocks.txt

  Scenario: API key not hardcoded in tests
    Tool: Bash (grep)
    Preconditions: test file exists
    Steps:
      1. Run: grep "tvly-" tests/test_tavily_tools.py
    Expected Result: No match (key read from env, not in source)
    Failure Indicators: Real key found in test file
    Evidence: .sisyphus/evidence/task-5-no-hardcoded-key.txt
  ```

  **Commit**: YES (separate commit)
  - Message: `test(tools): add integration tests for tavily tools`
  - Files: `tests/test_tavily_tools.py`
  - Pre-commit: `pytest tests/test_tavily_tools.py -v`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
>
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `pytest tests/test_tavily_tools.py -v`. Review `engine/tools/custom/tavily.py` for: `as any`/type ignores, empty catches, print in prod, commented-out code, unused imports. Check AI slop: excessive comments, over-abstraction, generic names. Verify all comments are in English (per AGENTS.md rule). Check that API key is not hardcoded anywhere.
  Output: `Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state. Run `pytest tests/test_tavily_tools.py -v`. Verify both tools are discoverable: `python -c "from engine import _discover_custom_tools; tools = _discover_custom_tools(); print([t.name for t in tools])"`. Check tool schemas are valid OpenAI function calling format. Save to `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Discovery [PASS/FAIL] | Schema [PASS/FAIL] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff. Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance: no modifications to config.py, tools/base.py, no new dependencies in pyproject.toml. Verify no tavily SDK imports anywhere.
  Output: `Tasks [N/N compliant] | Creep [CLEAN/N issues] | VERDICT`

---

## Commit Strategy

- **Commit 1**: `feat(tools): add tavily client and search/extract tools` — `engine/tools/custom/tavily.py`, `.env.example`
- **Commit 2**: `test(tools): add integration tests for tavily tools` — `tests/test_tavily_tools.py`

---

## Success Criteria

### Verification Commands
```bash
# Tools are auto-discovered
python -c "from engine import _discover_custom_tools; tools = _discover_custom_tools(); names = [t.name for t in tools]; assert 'tavily_search' in names; assert 'tavily_extract' in names; print('OK:', names)"

# Tests pass
pytest tests/test_tavily_tools.py -v

# No tavily SDK dependency
grep -r "tavily" pyproject.toml  # Expected: no match

# No core file modifications
git diff engine/config.py engine/tools/base.py  # Expected: empty
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass
- [ ] No new dependencies added
- [ ] No core files modified
- [ ] All comments in English
