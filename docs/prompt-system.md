# Prompt System Layered Redesign

## TL;DR

> **Quick Summary**: Refactor the flat `engine/prompts.py` into a layered `engine/prompts/` folder with 4-layer system prompt assembly: Identity/Rules → Environment → Tool Awareness → Custom Instructions.
>
> **Deliverables**:
> - `engine/prompts/` folder with `base.md`, `spawn.md`, `subagent.md`, `builder.py`, `env_builder.py`, `runtime.py`, `__init__.py`
> - Rewritten `base.md` without role assignment
> - Markdown-format Environment Context (replacing XML `<env>`)
> - Tool `short_description` attribute injected as prompt section
> - FRIDAY.md user instructions scanning and injection
> - `workspace_dir` config with auto-creation
> - Updated `engine.json.example` and `docs/codebase-structure.md`
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Task 1 (scaffolding) → Task 2-5 (parallel) → Task 6 (integration) → Task 7 (config) → Task 8 (docs)

---

## Context

### Original Request
Redesign the agent system prompt architecture, moving from a single flat `prompts.py` to a layered folder structure. Research was conducted on OpenCode and OpenClaw prompt systems to inform design decisions.

### Interview Summary
**Key Discussions**:
- BASE_PROMPT must NOT assign a role — only behavioral constraints and strategy
- Environment Context switches from XML `<env>` to Markdown `## Environment`
- Tool info injected via `short_description` attribute on each Tool class
- User custom instructions stored in FRIDAY.md alongside engine.json
- Working directory defaults to `project_root/workspace/`, configurable via config, auto-created
- Child agent results stay as user message injection (no change)
- Runtime directives (summary warning, child results) stay as Python functions in `runtime.py`

**Research Findings**:
- OpenCode: 6-layer stack, no tool info in prompt, AGENTS.md upward scanning
- OpenClaw: 30+ sections, tool summaries in prompt, context file ordering with stable/dynamic split
- Both use user messages for async child results
- Working directory always communicated via system prompt, never in tool schemas

### Metis Review
**Identified Gaps** (addressed):
- Config doesn't store engine.json directory path — need `config_dir` field for FRIDAY.md scanning
- `_refresh_env_block()` regex targets `<env>` XML — must switch to Markdown anchor
- `delegate()` system prompt assembly is inline — must route through new builder
- `Tool` base class has no `short_description` — need new class attribute with fallback
- Import paths for `prompts.py` consumers must be updated (`engine.prompts` → `engine.prompts` same path, different module type)

---

## Work Objectives

### Core Objective
Replace the flat `engine/prompts.py` with a layered `engine/prompts/` package that assembles system prompts from independent layers, while preserving backward compatibility.

### Concrete Deliverables
- `engine/prompts/` directory with all prompt files
- `engine/prompts/base.md` — behavioral constraints without role assignment
- `engine/prompts/spawn.md` — sub-agent spawning strategy
- `engine/prompts/subagent.md` — sub-agent system prompt template
- `engine/prompts/builder.py` — layer assembly logic
- `engine/prompts/runtime.py` — existing dynamic prompt functions
- `engine/prompts/__init__.py` — public API re-exports
- Updated `engine/tools/base.py` — `short_description` on Tool class
- Updated `engine/config.py` — `workspace_dir`, `config_dir` fields
- Updated `engine/runner.py` — route through new builder, FRIDAY.md scanning
- Updated `engine/time.py` — Markdown format for env block
- Updated `engine.json.example`
- Updated `docs/codebase-structure.md`

### Definition of Done
- [ ] `from engine.prompts import build_system_prompt, DEFAULT_SYSTEM_PROMPT` still works
- [ ] Generated system prompt contains all 4 layers in correct order
- [ ] FRIDAY.md content appears in prompt when file exists, absent when it doesn't
- [ ] Tool short_descriptions appear in prompt for enabled tools only
- [ ] Environment block is Markdown format (not XML)
- [ ] No role assignment in base.md
- [ ] All existing tests pass without modification
- [ ] docs/codebase-structure.md reflects new structure

### Must Have
- Layered prompt assembly with clear separation
- base.md without role assignment, with "prioritize user-defined role" clause
- Markdown environment context
- Tool short_description injection
- FRIDAY.md scanning from engine.json directory
- workspace_dir config with auto-creation
- Backward-compatible public API (same import paths, same function signatures)

### Must NOT Have (Guardrails)
- Do NOT add skills system
- Do NOT add provider-specific prompt variants
- Do NOT add cache boundary optimization
- Do NOT change child result injection mechanism (stays user message)
- Do NOT redesign sub-agent prompt (migrate existing template as-is to subagent.md)
- Do NOT add global FRIDAY.md (only project-level, same dir as engine.json)
- Do NOT assign any role/persona in base.md
- Do NOT put working directory info in tool descriptions

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (pytest + pytest-asyncio)
- **Automated tests**: Tests-after
- **Framework**: pytest

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Module/Config**: Use Bash (Python REPL / pytest) — Import, call functions, verify output

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately - scaffolding + config prerequisites):
├── Task 1: Create prompts/ folder structure + migrate runtime.py [quick]
├── Task 2: Add config fields (workspace_dir, config_dir) [quick]
└── Task 3: Add short_description to Tool base class [quick]

Wave 2 (After Wave 1 - core content, MAX PARALLEL):
├── Task 4: Write base.md (depends: 1) [quick]
├── Task 5: Write spawn.md (depends: 1) [quick]
├── Task 6: Migrate subagent.md template (depends: 1) [quick]
├── Task 7: Rewrite builder.py assembly logic (depends: 1, 2, 3) [unspecified-high]
└── Task 8: Create env_builder.py + update time.py (depends: 1) [unspecified-high]

Wave 3 (After Wave 2 - integration):
├── Task 9: Update runner.py to use new builder (depends: 4, 5, 7, 8) [unspecified-high]
└── Task 10: Update engine.json.example + docs (depends: 2, 9) [quick]

Wave FINAL (After ALL tasks — reviews):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
└── Task F3: Scope fidelity check (unspecified-high)

Critical Path: Task 1 → Task 7 → Task 9 → Task 10 → F1-F3
Parallel Speedup: ~60% faster than sequential
Max Concurrent: 5 (Wave 2)
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|-----------|--------|------|
| 1 | - | 4, 5, 6, 7, 8 | 1 |
| 2 | - | 7, 9, 10 | 1 |
| 3 | - | 7, 9 | 1 |
| 4 | 1 | 9 | 2 |
| 5 | 1 | 9 | 2 |
| 6 | 1 | 9 | 2 |
| 7 | 1, 2, 3 | 9 | 2 |
| 8 | 1 | 9 | 2 |
| 9 | 4, 5, 7, 8 | 10 | 3 |
| 10 | 2, 9 | F1-F3 | 3 |

### Agent Dispatch Summary

- **Wave 1**: 3 tasks — all `quick`
- **Wave 2**: 5 tasks — 3× `quick`, 2× `unspecified-high`
- **Wave 3**: 2 tasks — 1× `unspecified-high`, 1× `quick`
- **FINAL**: 3 tasks — 1× `oracle`, 2× `unspecified-high`

---

## TODOs

- [ ] 1. Create prompts/ package scaffold and migrate runtime.py

  **What to do**:
  - Create `engine/prompts/` directory
  - Move all dynamic prompt functions from current `engine/prompts.py` into `engine/prompts/runtime.py`:
    - `get_summary_warning()`
    - `get_emergency_summary_prompt()`
    - `get_child_results_prompt()`
    - `get_child_results_empty_warning()`
    - `get_spawn_confirmation()`
    - `get_concurrency_timeout_rejection()`
    - `get_runtime_depth_rejection()`
  - Move `DEPTH_LIMIT_REJECTION` constant to `runtime.py`
  - Create `engine/prompts/__init__.py` with re-exports of all public symbols (maintaining backward compatibility)
  - Note: `build_root_system_prompt` is removed — do NOT re-export it. The only caller (`runner.py`) is updated in Task 9 to use `build_system_prompt()` directly.
  - Create empty `engine/prompts/builder.py` (placeholder, filled in Task 7)
  - Create empty `engine/prompts/env_builder.py` (placeholder, filled in Task 8)
  - Create empty `engine/prompts/base.md` (placeholder, filled in Task 4)
  - Create empty `engine/prompts/spawn.md` (placeholder, filled in Task 5)
  - Create empty `engine/prompts/subagent.md` (placeholder, filled in Task 6)
  - Delete the old `engine/prompts.py` file
  - Verify `from engine.prompts import DEPTH_LIMIT_REJECTION, get_summary_warning` still works

  **Must NOT do**:
  - Do NOT change function signatures or behavior
  - Do NOT change import paths that consumers use (`engine.prompts.X` stays the same)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 2, 3)
  - **Parallel Group**: Wave 1
  - **Blocks**: Tasks 4, 5, 6, 7, 8
  - **Blocked By**: None

  **References**:
  - `engine/prompts.py` — current flat module to decompose (ALL content must be migrated)
  - `engine/__init__.py` — re-exports `DEFAULT_SYSTEM_PROMPT` from `engine.prompts`, verify it still resolves
  - `engine/runner.py:24` — `from engine.prompts import build_root_system_prompt` import (will be replaced with `build_system_prompt` in Task 9)
  - `engine/subagent/manager.py` — uses `get_spawn_confirmation`, `get_concurrency_timeout_rejection`, `get_child_results_prompt`, `get_child_results_empty_warning`
  - `engine/subagent/spawn.py` — uses `DEPTH_LIMIT_REJECTION`, `get_runtime_depth_rejection`
  - `engine/runtime/agent.py` — uses `get_summary_warning`, `get_emergency_summary_prompt`

  **Acceptance Criteria**:
  - [ ] `engine/prompts/` directory exists with `__init__.py`, `builder.py`, `env_builder.py`, `runtime.py`, `base.md`, `spawn.md`, `subagent.md`
  - [ ] Old `engine/prompts.py` deleted
  - [ ] `from engine.prompts import DEPTH_LIMIT_REJECTION` works
  - [ ] `from engine.prompts import get_summary_warning` works

  **QA Scenarios**:
  ```
  Scenario: Backward-compatible imports still work
    Tool: Bash (python -c)
    Preconditions: engine/prompts/ package exists, old prompts.py deleted
    Steps:
      1. Run: uv run python -c "from engine.prompts import DEPTH_LIMIT_REJECTION; print(DEPTH_LIMIT_REJECTION)"
      2. Run: uv run python -c "from engine.prompts import get_summary_warning; print(get_summary_warning(3))"
      3. Run: uv run python -c "from engine import DEFAULT_SYSTEM_PROMPT; print(len(DEFAULT_SYSTEM_PROMPT))"
    Expected Result: All three commands succeed without ImportError
    Failure Indicators: ImportError, ModuleNotFoundError, AttributeError
    Evidence: .sisyphus/evidence/task-1-import-check.txt
  ```

  **Commit**: YES (groups with Tasks 2, 3)
  - Message: `refactor(prompts): scaffold prompts package with config and tool prerequisites`
  - Files: `engine/prompts/`, `engine/prompts.py` (deleted)

- [ ] 2. Add config fields: workspace_dir and config_dir

  **What to do**:
  - Add to `Config` dataclass in `engine/config.py`:
    - `workspace_dir: Optional[str] = None` — user-configured workspace directory
  - In `ConfigLoader.load_from_json()`, after `find_config_file()` resolves the path:
    - Store the directory of engine.json on the Config object as `config_dir: str`
    - This is needed for FRIDAY.md scanning later
  - Add `workspace_dir` to the JSON loading logic in `load_from_json()`
  - Add workspace_dir resolution logic: if not configured, default to `{config_dir}/workspace/`

  **Must NOT do**:
  - Do NOT create the workspace directory yet (that happens in runner.py integration)
  - Do NOT break existing config loading

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 1, 3)
  - **Parallel Group**: Wave 1
  - **Blocks**: Tasks 7, 9, 10
  - **Blocked By**: None

  **References**:
  - `engine/config.py` — `Config` dataclass (add fields) and `ConfigLoader` (add loading logic)
  - `engine/config.py:ConfigLoader.find_config_file()` — returns the engine.json path, use `Path(...).parent` for config_dir
  - `engine/config.py:ConfigLoader.load_from_json()` — where JSON parsing happens, add new fields here
  - `engine.json.example` — add `workspace_dir` example

  **Acceptance Criteria**:
  - [ ] `Config` has `workspace_dir` and `config_dir` fields
  - [ ] `config_dir` is populated from engine.json parent directory
  - [ ] Default `workspace_dir` is `{config_dir}/workspace/`
  - [ ] Existing config loading still works

  **QA Scenarios**:
  ```
  Scenario: Config loads new fields correctly
    Tool: Bash (python -c)
    Preconditions: engine.json exists
    Steps:
      1. Run: uv run python -c "from engine.config import get_config; c = get_config(); print(f'config_dir={c.config_dir}, workspace_dir={c.workspace_dir}')"
    Expected Result: config_dir prints the engine.json parent directory, workspace_dir defaults to {config_dir}/workspace/
    Failure Indicators: AttributeError, config_dir is None
    Evidence: .sisyphus/evidence/task-2-config-fields.txt
  ```

  **Commit**: YES (groups with Tasks 1, 3)
  - Message: `refactor(prompts): scaffold prompts package with config and tool prerequisites`

- [ ] 3. Add short_description to Tool base class

  **What to do**:
  - In `engine/tools/base.py`, add `short_description: Optional[str] = None` as a class attribute on `Tool`
  - For `FunctionTool`, propagate `short_description` from the wrapped function or accept it as a constructor parameter
  - Add `short_description` to existing tool implementations:
    - `engine/tools/custom/web_search.py` (if it exists) — e.g. "Search the web using multiple search engines with auto-failover"
    - `engine/tools/custom/web_fetch.py` — e.g. "Fetch and convert web pages to markdown or text"
  - The `SpawnTool` in `engine/subagent/spawn.py` — e.g. "Delegate subtasks to child agents for parallel execution"
  - Ensure backward compatibility: tools without `short_description` simply return None

  **Must NOT do**:
  - Do NOT change tool execution behavior
  - Do NOT add short_description to tool JSON schema (it's for prompt only, not API)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 1, 2)
  - **Parallel Group**: Wave 1
  - **Blocks**: Tasks 7, 9
  - **Blocked By**: None

  **References**:
  - `engine/tools/base.py` — `Tool` ABC and `FunctionTool` class definitions
  - `engine/tools/custom/web_search.py` — existing custom tool to add short_description
  - `engine/tools/custom/web_fetch.py` — existing custom tool to add short_description
  - `engine/subagent/spawn.py` — `SpawnTool` class to add short_description
  - `engine/tools/pack.py` — `ToolPack` may need a method to collect short_descriptions

  **Acceptance Criteria**:
  - [ ] `Tool` class has `short_description` attribute
  - [ ] `FunctionTool` propagates `short_description`
  - [ ] All existing tools have meaningful `short_description` values
  - [ ] Tools without short_description don't break

  **QA Scenarios**:
  ```
  Scenario: Tool short_description accessible
    Tool: Bash (python -c)
    Preconditions: Tool class updated
    Steps:
      1. Run: uv run python -c "from engine.tools.base import Tool, FunctionTool; assert hasattr(Tool, 'short_description')"
      2. Run: uv run python -c "from engine.subagent.spawn import SpawnTool; t = SpawnTool(); print(t.short_description)"
    Expected Result: SpawnTool has a non-None short_description string
    Failure Indicators: AttributeError, None value
    Evidence: .sisyphus/evidence/task-3-tool-short-desc.txt
  ```

  **Commit**: YES (groups with Tasks 1, 2)
  - Message: `refactor(prompts): scaffold prompts package with config and tool prerequisites`

- [ ] 4. Write base.md — behavioral constraints without role assignment

  **What to do**:
  - Write `engine/prompts/base.md` with the EXACT content below
  - No role assignment — only behavioral constraints and execution strategy
  - Includes user role priority clause

  **Exact content for `engine/prompts/base.md`:**

  ```markdown
  # Execution Strategy

  1. **Use tools proactively** — When tools are available, prefer using them over reasoning from incomplete knowledge. Vary your approach if a tool returns weak or empty results.
  2. **Ground responses in evidence** — Strictly base answers and next actions on tool results. Never fabricate information or speculate beyond what the evidence supports.
  3. **Verify before finalizing** — For code or artifacts, prefer the smallest meaningful verification step: test, typecheck, lint, build, or direct inspection.

  # Output Format

  When the task specifies an output format, follow it exactly. The guidelines below apply when no format is specified.

  - Start with the direct answer or conclusion.
  - Follow with supporting details only when they add value.
  - No filler, no meta-commentary ("I have completed...", "Here is...").
  - For multi-part tasks, use clear headings or bullet lists.

  # Custom Instructions Priority

  If the user provides a role definition, persona, or additional behavioral instructions, prioritize following those. User-defined instructions override the default strategy above whenever they conflict.
  ```

  **Must NOT do**:
  - Do NOT include any role assignment like "You are an assistant" or "You are the root agent"
  - Do NOT include spawning strategy (that's in spawn.md)
  - Do NOT include environment info (that's Layer 2)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 5, 6, 7, 8)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 9
  - **Blocked By**: Task 1

  **References**:
  - Current `engine/prompts.py:BASE_PROMPT` — content to replace (current version starts with "You are the root orchestrator agent" — this line and any role assignment must be removed)
  - OpenClaw's `<execution_policy>` and `<tool_discipline>` in `src/agents/gpt5-prompt-overlay.ts` — reference for behavioral constraint style (concise, no role)
  - OpenCode's `session/prompt/default.txt` — reference for concise constraint style

  **Acceptance Criteria**:
  - [ ] `base.md` exists with the exact content specified above
  - [ ] Does NOT contain "You are" or any role/persona assignment
  - [ ] Contains "Use tools proactively" clause
  - [ ] Contains "Custom Instructions Priority" clause
  - [ ] Contains "Ground responses in evidence" clause

  **QA Scenarios**:
  ```
  Scenario: base.md has no role assignment
    Tool: Bash (grep)
    Preconditions: base.md written
    Steps:
      1. Grep base.md for "You are" pattern — should find zero matches
      2. Grep base.md for "Custom Instructions Priority" — should find exactly 1 match
      3. Read base.md and verify it's under 30 lines
    Expected Result: No role assignment, contains behavioral constraints and user priority clause
    Failure Indicators: "You are the" or "You are an" found in file
    Evidence: .sisyphus/evidence/task-4-base-check.txt
  ```

  **Commit**: YES (groups with Tasks 5, 6)
  - Message: `refactor(prompts): add prompt content files (base, spawn, subagent)`

- [ ] 5. Write spawn.md — sub-agent spawning strategy

  **What to do**:
  - Write `engine/prompts/spawn.md` with the spawning strategy
  - Migrate content from current `engine/prompts.py:SPAWN_PROMPT`
  - Keep as-is: decomposition-first, parallel over sequential, handle simple tasks yourself, iterate after synthesis, spawning rules
  - No changes needed to content, just move to markdown file

  **Must NOT do**:
  - Do NOT change the spawning strategy semantics
  - Do NOT add role assignment

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 4, 6, 7, 8)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 9
  - **Blocked By**: Task 1

  **References**:
  - Current `engine/prompts.py:SPAWN_PROMPT` — exact content to migrate

  **Acceptance Criteria**:
  - [ ] `spawn.md` exists with spawning strategy content
  - [ ] Content matches current SPAWN_PROMPT semantics

  **QA Scenarios**:
  ```
  Scenario: spawn.md content preserved
    Tool: Bash (python -c)
    Preconditions: spawn.md written
    Steps:
      1. Read spawn.md, verify it contains "Decompose first" and "Parallel over sequential"
    Expected Result: Key spawning rules present in markdown file
    Failure Indicators: Missing key rules
    Evidence: .sisyphus/evidence/task-5-spawn-check.txt
  ```

  **Commit**: YES (groups with Tasks 4, 6)
  - Message: `refactor(prompts): add prompt content files (base, spawn, subagent)`

- [ ] 6. Migrate subagent.md — sub-agent system prompt template

  **What to do**:
  - Write `engine/prompts/subagent.md` with the sub-agent prompt template
  - Migrate content from current `engine/prompts.py:get_subagent_system_prompt()` template string
  - Use `string.Template` syntax: `$variable` for placeholders (NOT Python `{variable}` `.format()` syntax)
    - Reason: `.format()` breaks if the markdown content contains literal `{}` (e.g., JSON examples, code blocks)
    - `string.Template` only recognizes `$variable` and `${variable}`, making it safe for markdown content
  - Keep the template as close to current as possible — this is a migration, not a redesign

  **Placeholder mapping (current → new):**
  - `{parent_label}` → `$parent_label`
  - `{task_desc}` → `$task_desc`
  - `{depth}` → `$depth`
  - `{max_depth}` → `$max_depth`
  - `{spawn_section}` → `$spawn_section`
  - `{task_id}` → `$task_id`
  - `{label}` → `$label`

  **Must NOT do**:
  - Do NOT redesign the sub-agent prompt structure
  - Do NOT use Python `.format()` placeholders `{variable}` — use `$variable` instead

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 4, 5, 7, 8)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 9
  - **Blocked By**: Task 1

  **References**:
  - Current `engine/prompts.py:get_subagent_system_prompt()` — the template string to migrate (lines 134-184)
  - `engine/subagent/manager.py` — the caller that uses this template
  - Python `string.Template` docs — the safe templating mechanism to use

  **Acceptance Criteria**:
  - [ ] `subagent.md` exists with template content
  - [ ] Contains all `$variable` placeholders: `$parent_label`, `$task_desc`, `$depth`, `$max_depth`, `$spawn_section`, `$task_id`, `$label`
  - [ ] Does NOT contain Python `.format()` style `{variable}` placeholders

  **QA Scenarios**:
  ```
  Scenario: subagent.md template works with string.Template
    Tool: Bash (python -c)
    Preconditions: subagent.md written
    Steps:
      1. Read subagent.md, verify it contains $parent_label, $task_desc, $depth, $max_depth
      2. Run: python -c "from string import Template; t = Template(open('engine/prompts/subagent.md').read()); result = t.substitute(parent_label='root', task_desc='test', depth=1, max_depth=3, spawn_section='leaf', task_id='t1', label='child'); print(len(result))"
      3. Verify no KeyError or ValueError
    Expected Result: Template substitutes correctly, all placeholders resolved
    Failure Indicators: KeyError, ValueError, $variable still present in output
    Evidence: .sisyphus/evidence/task-6-subagent-check.txt
  ```

  **Commit**: YES (groups with Tasks 4, 5)
  - Message: `refactor(prompts): add prompt content files (base, spawn, subagent)`

- [ ] 7. Write builder.py — layer assembly logic

  **What to do**:
  - Implement `engine/prompts/builder.py` with the core assembly logic below

  **Core code for `engine/prompts/builder.py`:**

  ```python
  """Prompt layer assembly for the engine framework."""

  from pathlib import Path
  from string import Template
  from typing import Dict, List, Optional, Tuple

  _PROMPTS_DIR = Path(__file__).parent


  def _read_md(filename: str) -> str:
      """Read a markdown file from the prompts directory."""
      return (_PROMPTS_DIR / filename).read_text().strip()


  def build_system_prompt(
      include_spawn: bool = False,
      env_context: Optional[Dict[str, str]] = None,
      tool_descriptions: Optional[List[Tuple[str, str]]] = None,
      user_instructions: Optional[str] = None,
  ) -> str:
      """Assemble the full system prompt from all layers.

      Layer order:
        1. base.md (+ optional spawn.md)
        2. ## Environment (Markdown key-value)
        3. ## Available Tools (tool short_descriptions)
        4. ## Custom Instructions (FRIDAY.md content)
      """
      sections = [_read_md("base.md")]

      if include_spawn:
          sections.append(_read_md("spawn.md"))

      if env_context:
          lines = ["## Environment"]
          for key, value in env_context.items():
              lines.append(f"- **{key}**: {value}")
          sections.append("\n".join(lines))

      if tool_descriptions:
          lines = ["## Available Tools"]
          for name, desc in tool_descriptions:
              lines.append(f"- **{name}**: {desc}")
          sections.append("\n".join(lines))

      if user_instructions:
          sections.append(f"## Custom Instructions\n\n{user_instructions.strip()}")

      return "\n\n".join(sections)


  DEFAULT_SYSTEM_PROMPT: str = build_system_prompt(include_spawn=True)
  # build_root_system_prompt() removed — redundant with build_system_prompt()
  # When all optional params are None, build_system_prompt() returns base + optional spawn


  def get_subagent_system_prompt(
      parent_label: str,
      task_desc: str,
      depth: int,
      max_depth: int,
      can_spawn: bool,
      task_id: str,
      label: str = "",
  ) -> str:
      """Build the raw sub-agent template (no env layer).

      Reads from subagent.md which uses $variable placeholders
      processed via string.Template (safe for markdown content).
      """
      spawn_section = (
          "You CAN spawn your own sub-agents."
          if can_spawn
          else "You are a leaf worker and CANNOT spawn further sub-agents."
      )
      template = Template(_read_md("subagent.md"))
      return template.substitute(
          parent_label=parent_label,
          task_desc=task_desc,
          depth=depth,
          max_depth=max_depth,
          spawn_section=spawn_section,
          task_id=task_id,
          label=label,
      )


  def build_subagent_prompt(
      parent_label: str,
      task_desc: str,
      depth: int,
      max_depth: int,
      can_spawn: bool,
      task_id: str,
      label: str = "",
      env_context: Optional[Dict[str, str]] = None,
  ) -> str:
      """Assemble the full sub-agent system prompt with env layer.

      Called by SubAgentManager.spawn() — this is the single entry point
      for sub-agent prompt assembly. Keeps env formatting logic in one place
      (builder.py) instead of duplicated in manager.py.

      Layers:
        1. subagent.md template (variable-substituted)
        2. ## Environment (if env_context provided)
      """
      sections = [get_subagent_system_prompt(
          parent_label=parent_label,
          task_desc=task_desc,
          depth=depth,
          max_depth=max_depth,
          can_spawn=can_spawn,
          task_id=task_id,
          label=label,
      )]

      if env_context:
          lines = ["## Environment"]
          for key, value in env_context.items():
              lines.append(f"- **{key}**: {value}")
          sections.append("\n".join(lines))

      return "\n\n".join(sections)
  ```

  **Assembly logic summary:**
  - Each layer is a string section, joined with `\n\n`
  - `build_system_prompt()` is the single entry point for root agent — all optional params default to None
  - `DEFAULT_SYSTEM_PROMPT = build_system_prompt(include_spawn=True)` — Layer 1 only (backward compat)
  - `build_subagent_prompt()` assembles sub-agent prompt (template + env) — called by `SubAgentManager.spawn()` in `manager.py`
  - `get_subagent_system_prompt()` is the raw template substitution (called internally by `build_subagent_prompt`, also exposed for backward compat)
  - `build_root_system_prompt()` was REMOVED — redundant with `build_system_prompt()` when optional params are None

  **Must NOT do**:
  - Do NOT hardcode prompt text in Python — all content comes from .md files
  - Do NOT include environment/tool/user layers in `DEFAULT_SYSTEM_PROMPT` (those are runtime-only)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 4, 5, 6, 8)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 9
  - **Blocked By**: Tasks 1, 2, 3

  **References**:
  - Current `engine/prompts.py:build_root_system_prompt()` — REMOVED (redundant). Callers updated in Task 9 to use `build_system_prompt()`
  - Current `engine/prompts.py:get_subagent_system_prompt()` — template function to migrate
  - Current `engine/prompts.py:DEFAULT_SYSTEM_PROMPT` — backward-compatible constant
  - Python `string.Template` — safe template syntax for .md files with potential `{` `}` characters
  - OpenCode's `session/prompt.ts` — reference for layered join pattern
  - OpenClaw's `system-prompt.ts:buildAgentSystemPrompt()` — reference for section-by-section assembly

  **Acceptance Criteria**:
  - [ ] `build_system_prompt()` assembles all 4 layers in correct order
  - [ ] `build_system_prompt(include_spawn=True)` returns base + spawn content (replaces removed `build_root_system_prompt`)
  - [ ] `DEFAULT_SYSTEM_PROMPT` equals `build_system_prompt(include_spawn=True)`
  - [ ] `get_subagent_system_prompt()` reads from subagent.md via `string.Template`
  - [ ] Missing layers (no env, no tools, no user instructions) produce valid output without errors

  **QA Scenarios**:
  ```
  Scenario: Builder assembles all layers
    Tool: Bash (python -c)
    Preconditions: builder.py, base.md, spawn.md exist
    Steps:
      1. Call build_system_prompt(include_spawn=True, env_context={"Date": "Mon 2026", "Working Directory": "/tmp"}, tool_descriptions=[("spawn", "Delegate tasks")], user_instructions="Use Chinese")
      2. Verify output contains base.md content
      3. Verify output contains "## Environment"
      4. Verify output contains "## Available Tools"
      5. Verify output contains "## Custom Instructions"
      6. Verify order: base → spawn → Environment → Tools → Custom Instructions
    Expected Result: All 4 layers present in correct order
    Failure Indicators: Missing sections, wrong order
    Evidence: .sisyphus/evidence/task-7-builder-check.txt

  Scenario: Backward compatibility
    Tool: Bash (python -c)
    Steps:
      1. from engine.prompts import DEFAULT_SYSTEM_PROMPT, build_system_prompt
      2. assert DEFAULT_SYSTEM_PROMPT == build_system_prompt(include_spawn=True)
      3. assert "You are" NOT in DEFAULT_SYSTEM_PROMPT (no role assignment)
    Expected Result: Backward-compatible API works, no role in output
    Evidence: .sisyphus/evidence/task-7-backward-compat.txt

  Scenario: Sub-agent template via string.Template
    Tool: Bash (python -c)
    Steps:
      1. from engine.prompts import get_subagent_system_prompt
      2. result = get_subagent_system_prompt(parent_label="root", task_desc="search files", depth=1, max_depth=3, can_spawn=False, task_id="t1", label="searcher")
      3. Verify result contains "root" and "search files"
      4. Verify "$variable" NOT in result (all substituted)
    Expected Result: Template substitutes all $variable placeholders correctly
    Evidence: .sisyphus/evidence/task-7-subagent-template.txt
  ```

  **Commit**: YES (groups with Task 8)
  - Message: `refactor(prompts): implement builder and markdown env format`

- [ ] 8. Create env_builder.py + update time.py

  **What to do**:

  **Part A: Create `engine/prompts/env_builder.py`**

  New module dedicated to collecting environment context from multiple sources.
  This is NOT part of `TimeProvider` — env context is multi-source (time, config, system).

  ```python
  # engine/prompts/env_builder.py
  """Environment context builder for system prompt injection.

  Collects context from multiple sources (time, config, system)
  into a flat dict for the builder's ## Environment layer.
  """

  import platform
  from datetime import datetime
  from typing import Dict, Optional
  from zoneinfo import ZoneInfo


  def build_env_context(
      time_provider,                    # TimeProvider instance
      workspace_dir: str,               # resolved from Config
      model_name: str,                  # primary model label
      platform_override: Optional[str] = None,  # for testing
  ) -> Dict[str, str]:
      """Collect environment context from multiple sources.

      Returns a flat dict suitable for builder.py's ## Environment layer.
      Future fields can be added here without changing callers.
      """
      tz_name = time_provider.resolve_timezone()
      tz_obj = ZoneInfo(tz_name)
      date_str = datetime.now(tz=tz_obj).strftime("%a %b %d %Y")

      return {
          "Date": date_str,                   # e.g. "Tue May 05 2026"
          "Timezone": tz_name,                # e.g. "Asia/Shanghai"
          "Working Directory": workspace_dir,  # e.g. "/Users/sys/project/workspace"
          "Model": model_name,                # e.g. "gpt-4o"
          "OS": platform_override or platform.system(),  # e.g. "Darwin"
      }
  ```

  **Part B: Update `engine/time.py`**

  - **DELETE** `format_system_env_block()` method from `TimeProvider`
    - Env assembly is now `env_builder.build_env_context()` + `builder.build_system_prompt(env_context=...)`
    - `TimeProvider` reverts to pure time utility: `resolve_timezone()`, `format_message_timestamp()`, `inject_timestamp()`
  - Keep all other methods unchanged

  **Part C: Update `_refresh_env_block()` in `engine/runner.py`**

  - Change `_ENV_BLOCK_PATTERN` regex from XML `<env>...</env>` to Markdown `## Environment` heading
  - Update `_refresh_env_block()` to:
    1. Build fresh env dict via `env_builder.build_env_context(time_provider, workspace_dir, model_name)`
    2. Format as Markdown section
    3. Replace the `## Environment` section in the existing system message

  **Must NOT do**:
  - Do NOT add env context fields to `TimeProvider` — it's a time utility only
  - Do NOT hardcode env fields in builder.py — env_builder.py owns field definitions
  - Do NOT break session reuse (env block refresh must still work)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 4, 5, 6, 7)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 9
  - **Blocked By**: Task 1

  **References**:
  - `engine/time.py:TimeProvider.format_system_env_block()` — DELETE this method
  - `engine/time.py:TimeProvider.resolve_timezone()` — KEEP, used by env_builder
  - `engine/runner.py:_ENV_BLOCK_PATTERN` — regex pattern to update for Markdown
  - `engine/runner.py:_refresh_env_block()` — function to update (use env_builder instead of time_provider directly)
  - `engine/subagent/manager.py:206` — `self._time_provider.format_system_env_block()` call to remove (Task 9 handles full migration)

  **Acceptance Criteria**:
  - [ ] `engine/prompts/env_builder.py` exists with `build_env_context()` function
  - [ ] `build_env_context()` returns dict with keys: Date, Timezone, Working Directory, Model, OS
  - [ ] `TimeProvider.format_system_env_block()` is DELETED
  - [ ] `TimeProvider.resolve_timezone()`, `format_message_timestamp()`, `inject_timestamp()` still work
  - [ ] `_refresh_env_block()` uses Markdown regex `## Environment` instead of XML `<env>`

  **QA Scenarios**:
  ```
  Scenario: env_builder collects all fields
    Tool: Bash (python -c)
    Steps:
      1. from engine.prompts.env_builder import build_env_context
      2. from engine.time import TimeProvider
      3. ctx = build_env_context(TimeProvider(), "/tmp/ws", "gpt-4o")
      4. assert set(ctx.keys()) == {"Date", "Timezone", "Working Directory", "Model", "OS"}
      5. assert ctx["Working Directory"] == "/tmp/ws"
      6. assert ctx["Model"] == "gpt-4o"
    Expected Result: All 5 fields present with correct values
    Evidence: .sisyphus/evidence/task-8-env-builder-fields.txt

  Scenario: TimeProvider no longer has format_system_env_block
    Tool: Bash (python -c)
    Steps:
      1. from engine.time import TimeProvider
      2. assert not hasattr(TimeProvider, "format_system_env_block")
    Expected Result: Method is deleted
    Evidence: .sisyphus/evidence/task-8-time-cleanup.txt

  Scenario: _refresh_env_block uses Markdown regex
    Tool: Bash (python -c)
    Preconditions: runner.py updated
    Steps:
      1. Verify _ENV_BLOCK_PATTERN matches "## Environment" not "<env>"
      2. Create a mock session with "## Environment\n- **Date**: old" in system message
      3. Call _refresh_env_block with fresh env context
      4. Verify Date value is updated in-place
    Expected Result: Env section updated with fresh values
    Evidence: .sisyphus/evidence/task-8-refresh-markdown.txt
  ```

  **Commit**: YES (groups with Task 7)
  - Message: `refactor(prompts): add env_builder and remove env from time_provider`

- [ ] 9. Update runner.py — integrate builder, FRIDAY.md scanning, workspace_dir

  **What to do**:
  - Replace inline prompt assembly in `delegate()` with calls to `builder.build_system_prompt()`
  - Build env context via `env_builder.build_env_context()`:
    - Pass `time_provider`, resolved `workspace_dir`, and `model_name` (from config)
    - Pass the returned dict to `build_system_prompt(env_context=...)`
  - Add FRIDAY.md scanning:
    - Check `{config.config_dir}/FRIDAY.md` for existence
    - If exists, read its content as user instructions input
  - Add workspace_dir resolution and auto-creation:
    - Resolve workspace directory from config (or default to `{config_dir}/workspace/`)
    - Create directory if it doesn't exist (`os.makedirs(exist_ok=True)`)
    - Pass workspace_dir to env_builder
  - Collect tool short_descriptions:
    - Iterate `enabled_tools` list, collect `(tool.name, tool.short_description)` for tools where `short_description is not None`
    - Pass to builder
  - Update `_refresh_env_block()` to use Markdown regex anchor
  - Ensure session reuse path still works (env refresh, system prompt preserved)
  - **Update `engine/subagent/manager.py`** to use `build_subagent_prompt()` + `env_builder` instead of manual env concatenation:
    - Replace lines 196-207:
      ```python
      # BEFORE:
      system_prompt = get_subagent_system_prompt(...)
      env_block = self._time_provider.format_system_env_block()
      system_prompt = f"{system_prompt}\n\n{env_block}"

      # AFTER:
      from engine.prompts.builder import build_subagent_prompt
      from engine.prompts.env_builder import build_env_context
      env_context = build_env_context(
          time_provider=self._time_provider,
          workspace_dir=resolved_workspace_dir,
          model_name=model_name,
      )
      system_prompt = build_subagent_prompt(
          parent_label=parent_label,
          task_desc=task_desc,
          depth=child_session.depth,
          max_depth=config.max_depth,
          can_spawn=can_spawn,
          task_id=task_id,
          label=display_name,
          env_context=env_context,
      )
      ```
    - This ensures env formatting (Markdown vs XML) and field collection are centralized in env_builder.py + builder.py

  **Must NOT do**:
  - Do NOT change the delegate() function signature
  - Do NOT change the session reuse behavior
  - Do NOT add FRIDAY.md scanning to the session reuse path (FRIDAY.md is read once at session creation)
  - Do NOT keep manual env block concatenation in manager.py
  - Do NOT call `time_provider.format_system_env_block()` anywhere — it was deleted in Task 8
  - Do NOT construct env_context dict inline — use `env_builder.build_env_context()` always

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (sequential)
  - **Blocks**: Task 10
  - **Blocked By**: Tasks 4, 5, 7, 8

  **References**:
  - `engine/runner.py:delegate()` — main function to modify (lines 100-272)
  - `engine/runner.py:_refresh_env_block()` — regex update for Markdown (lines 83-97)
  - `engine/runner.py:_ENV_BLOCK_PATTERN` — regex to update (line 77-80)
  - `engine/subagent/manager.py:spawn()` — replace manual env concatenation with `build_subagent_prompt()` (lines 196-209)
  - `engine/prompts/builder.py` — new builder to call (from Task 7), specifically `build_subagent_prompt()` and `build_system_prompt()`
  - `engine/prompts/env_builder.py` — new env context collector (from Task 8), `build_env_context()`
  - `engine/config.py:Config` — new fields: `config_dir`, `workspace_dir`

  **Acceptance Criteria**:
  - [ ] `delegate()` uses builder for prompt assembly
  - [ ] FRIDAY.md content appears in system prompt when file exists
  - [ ] workspace_dir is created if it doesn't exist
  - [ ] Working directory in env context reflects resolved workspace_dir
  - [ ] Tool short_descriptions appear in system prompt
  - [ ] Model info appears in env context
  - [ ] Session reuse still works (env refresh)
  - [ ] `manager.py` uses `build_subagent_prompt()` — NO manual env string concatenation
  - [ ] All existing tests pass

  **QA Scenarios**:
  ```
  Scenario: Full prompt assembly with all layers
    Tool: Bash (python -c)
    Preconditions: FRIDAY.md exists in project root
    Steps:
      1. Create a temporary FRIDAY.md with test content
      2. Run delegate() with a simple task (mocked LLM)
      3. Inspect the system message in the session
      4. Verify it contains: base content, ## Environment, ## Available Tools, ## Custom Instructions
    Expected Result: All 4 layers present in final system message
    Evidence: .sisyphus/evidence/task-9-full-assembly.txt

  Scenario: No FRIDAY.md — prompt still valid
    Tool: Bash (python -c)
    Preconditions: FRIDAY.md does NOT exist
    Steps:
      1. Ensure no FRIDAY.md in project directory
      2. Run delegate() with a simple task (mocked LLM)
      3. Inspect the system message
      4. Verify NO "## Custom Instructions" section
      5. Verify other layers are present
    Expected Result: 3 layers present, no Custom Instructions
    Evidence: .sisyphus/evidence/task-9-no-friday.txt

  Scenario: Workspace auto-creation
    Tool: Bash (python -c)
    Preconditions: workspace directory does not exist
    Steps:
      1. Run delegate() (mocked LLM)
      2. Check if workspace directory was created
      3. Verify it exists and is a directory
    Expected Result: workspace/ directory created automatically
    Evidence: .sisyphus/evidence/task-9-workspace-creation.txt

  Scenario: Sub-agent prompt assembled by builder (not manager)
    Tool: Bash (grep)
    Preconditions: Task 9 complete
    Steps:
      1. grep -n "format_system_env_block\|f\"{system_prompt}" engine/subagent/manager.py
      2. Verify NO results (manual env concatenation removed)
      3. grep -n "build_subagent_prompt" engine/subagent/manager.py
      4. Verify ONE result (using builder function)
    Expected Result: No manual env block concatenation in manager.py, only builder call
    Failure Indicators: `format_system_env_block` or `f"{system_prompt}` found in manager.py
    Evidence: .sisyphus/evidence/task-9-subagent-builder-check.txt
  ```

  **Commit**: YES
  - Message: `refactor(runner): integrate new prompt builder and FRIDAY.md scanning`

- [ ] 10. Update engine.json.example and docs/codebase-structure.md

  **What to do**:
  - Update `engine.json.example` to add:
    - `"workspace_dir": "./workspace"` (with comment explaining default behavior)
  - Update `docs/codebase-structure.md`:
    - Replace `engine/prompts.py` entry with `engine/prompts/` package breakdown
    - Document the 4-layer prompt architecture
    - Update directory tree to show prompts/ as a directory, not a file
    - Update module details for any changed files (runner.py, config.py, time.py, tools/base.py)
  - Update `AGENTS.md` if needed (no changes expected)

  **Must NOT do**:
  - Do NOT change actual code behavior
  - Do NOT commit if docs are out of sync with actual code

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (after Task 9)
  - **Blocks**: F1-F3
  - **Blocked By**: Tasks 2, 9

  **References**:
  - `engine.json.example` — add new config fields
  - `docs/codebase-structure.md` — comprehensive update needed
  - `AGENTS.md` — check if update needed

  **Acceptance Criteria**:
  - [ ] `engine.json.example` has `workspace_dir` field
  - [ ] `docs/codebase-structure.md` reflects new prompts/ package structure
  - [ ] Directory tree in docs shows prompts/ as folder with all files
  - [ ] Module details are accurate for changed files

  **QA Scenarios**:
  ```
  Scenario: Docs match code
    Tool: Bash (diff/grep)
    Steps:
      1. Verify docs/codebase-structure.md mentions engine/prompts/ as directory
      2. Verify docs mentions base.md, spawn.md, subagent.md, builder.py, runtime.py
      3. Verify engine.json.example has workspace_dir field
    Expected Result: Docs accurately reflect new code structure
    Evidence: .sisyphus/evidence/task-10-docs-check.txt
  ```

  **Commit**: YES
  - Message: `docs: update config example and codebase structure`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 3 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `uv run pytest`. Review all changed files for: type annotation issues, empty catches, commented-out code, unused imports. Check that `from engine.prompts import build_system_prompt` works. Verify `build_root_system_prompt` is NOT exported (removed). Verify no role assignment exists in base.md. Verify env block is Markdown not XML.
  Output: `Tests [PASS/FAIL] | Imports [PASS/FAIL] | No Role Assignment [PASS/FAIL] | Markdown Env [PASS/FAIL] | VERDICT`

- [ ] F3. **Scope Fidelity Check** — `unspecified-high`
  For each task: read "What to do", read actual diff. Verify 1:1 — everything in spec was built, nothing beyond spec was built. Check "Must NOT do" compliance. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **Task 1-3**: `refactor(prompts): scaffold prompts package with config and tool prerequisites`
- **Task 4-6**: `refactor(prompts): add prompt content files (base, spawn, subagent)`
- **Task 7-8**: `refactor(prompts): implement builder and markdown env format`
- **Task 9**: `refactor(runner): integrate new prompt builder and FRIDAY.md scanning`
- **Task 10**: `docs: update config example and codebase structure`

---

## Success Criteria

### Verification Commands
```bash
uv run pytest                                          # Expected: all tests pass
uv run python -c "from engine.prompts import build_system_prompt, DEFAULT_SYSTEM_PROMPT; print(build_system_prompt(include_spawn=True)[:50])"  # Expected: prints prompt start
uv run python -c "from engine.prompts import DEFAULT_SYSTEM_PROMPT; assert '## Environment' not in DEFAULT_SYSTEM_PROMPT"  # Expected: no env in static prompt
uv run python -c "from engine.tools.base import Tool; assert hasattr(Tool, 'short_description')"  # Expected: True
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All existing tests pass
- [ ] docs/codebase-structure.md updated
- [ ] No role assignment in base.md
- [ ] FRIDAY.md scanning works
- [ ] Tool short_descriptions in prompt
