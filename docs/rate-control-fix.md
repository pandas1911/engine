# 修复 Key Pool 排序 Bug + 清理 Dead Field

## TL;DR

> **Quick Summary**: 修复 `acquire_key()` 因 `consecutive_errors` 排序导致 primary 冷却后无法恢复优先级的 bug，同时清理 `ProviderHealth.pace_level` 死字段。
>
> **Deliverables**:
> - 修复后的 `key_pool.py`：primary 冷却到期后自动恢复第一优先级
> - 清理后的 `provider_models.py`：移除未使用的 `pace_level` 字段
> - 新增测试覆盖排序修复
> - 更新 `docs/codebase-structure.md`
>
> **Estimated Effort**: Quick
> **Parallel Execution**: NO - 2 个任务顺序执行（共享测试文件）
> **Critical Path**: Task 1 → Task 2

---

## Context

### Original Request

用户在分析日志 `run_20260509_161459.jsonl` 时发现 `key_cooldown` 问题。经深入代码分析，发现 `acquire_key()` 的排序逻辑打破了 primary → fallback 的优先级。同时发现 `ProviderHealth.pace_level` 是死代码。

### Interview Summary

**Key Discussions**:
- 分析了 pace_wait level=healthy 但 key_cooldown 发生的原因：本地 RPM 窗口和上游 API 限流阈值不一致
- 确认了 burst 限制的概念，但最终决定不加 burst limiter
- 讨论了 429 反馈回路（pace override），**用户指出在当前架构下无意义**——key cooldown 后直接切到 fallback provider，minimax 的 limiter 不会被访问直到 cooldown 结束，此时窗口自然冷却
- 确认 `ProviderHealth.pace_level` 删除

**Research Findings**:
- Metis 分析确认：Python `dict.items()` 在 3.7+ 保证插入顺序，删除 sort 后 primary 自然排第一
- Metis 确认：所有 `ProviderHealth` 构造都使用 keyword args，删除字段不影响调用方
- 现有测试 `test_fallback_truncation.py` 通过手动 cooldown 来测试 key 切换，排序修复不影响

### Metis Review

**Identified Gaps** (addressed):
- 429 反馈回路：用户指出在当前 key 切换架构下无意义，从计划中移除
- `consecutive_errors` 在排序修复后仍用于冷却阶梯计算，不受影响
- 所有 limiter 为 None 的场景已通过 `if limiter is not None` 模式保护

---

## Work Objectives

### Core Objective

确保 primary provider 在 cooldown 到期后自动恢复第一优先级，不受 `consecutive_errors` 影响。清理无用的死代码字段。

### Concrete Deliverables

- `engine/safety/key_pool.py`：修复 `acquire_key()` 排序逻辑
- `engine/providers/provider_models.py`：移除 `pace_level` 字段
- `tests/test_key_pool_sorting.py`（或追加到现有测试文件）：排序修复的测试
- `docs/codebase-structure.md`：同步更新

### Definition of Done

- [ ] `uv run pytest` 全部通过
- [ ] primary cooldown 到期后，即使 `consecutive_errors=1` 而 fallback `consecutive_errors=0`，primary 仍被选中
- [ ] `ProviderHealth` 不再包含 `pace_level` 字段
- [ ] `docs/codebase-structure.md` 与实际代码一致

### Must Have

- `acquire_key()` 不再用 `consecutive_errors` 排序，而是用插入顺序（primary 优先）
- cooldown 到期后 primary 自动恢复
- `consecutive_errors` 仍用于冷却阶梯计算（30s → 60s → 300s），不受影响
- "所有 key 都在 cooldown" 的 fallback 路径不受影响

### Must NOT Have (Guardrails)

- 不修改 `consecutive_errors` 的增减逻辑
- 不修改冷却阶梯计算（30s/60s/300s）
- 不添加 burst limiter
- 不添加 429 反馈回路到 rate_limiter
- 不修改 `_scheduler()`、waiter queue 或 `_wait_if_needed()` 逻辑
- 不改变 fallback_provider.py 的错误处理流程
- 不增加跨模块耦合（不传 limiter 引用到 key_pool）

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed.

### Test Decision

- **Infrastructure exists**: YES (pytest, uv run pytest)
- **Automated tests**: YES (tests-after)
- **Framework**: pytest with pytest-asyncio

### QA Policy

Every task includes agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Unit tests**: Use `uv run pytest` - run tests, assert pass/fail
- **Code verification**: Use Bash - read files, check content

---

## Execution Strategy

### Sequential Execution

```
Task 1: Fix acquire_key() sorting + update docstrings  [quick]
Task 2: Remove ProviderHealth.pace_level dead field     [quick]
```

### Dependency Matrix

- **1**: - → 2
- **2**: 1 → -

### Agent Dispatch Summary

- **Task 1**: `quick` — 单文件修改 + 测试
- **Task 2**: `quick` — 两个字段删除 + 测试

---

## TODOs

- [ ] 1. 修复 `acquire_key()` 排序逻辑 + 更新文档字符串

  **What to do**:
  - 在 `engine/safety/key_pool.py` 中：
    - 删除 `acquire_key()` 第 61 行的 `candidates.sort(key=lambda x: x[1].consecutive_errors)`
    - 直接 `return candidates[0][0]`（`dict.items()` 在 Python 3.7+ 保证插入顺序，primary 自然排第一）
    - 更新 `acquire_key()` 的 docstring（第 40~47 行）：移除 "prefers keys with the lowest consecutive_errors" 描述，改为明确说明 "Returns first available key in insertion order (primary first). consecutive_errors only affects cooldown duration, not selection priority."
    - 更新类 docstring（第 13~23 行）：移除第 21~22 行关于 "prefers the lowest consecutive_errors" 的描述
  - 新建或追加测试（`tests/test_key_pool_sorting.py` 或追加到 `tests/test_fallback_truncation.py`）：
    - **Test A**: primary 有 `consecutive_errors=1`，fallback 有 `consecutive_errors=0`，primary cooldown 已到期 → 断言 `acquire_key()` 返回 primary
    - **Test B**: 所有 key 都在 cooldown → 断言返回最早到期的 key（现有行为不变）
    - **Test C**: `report_success()` 仍正确重置 `consecutive_errors=0` 和 `cooldown_until=None`
    - **Test D**: 单 key pool 不受影响
  - 运行 `uv run pytest` 确认所有测试通过

  **Must NOT do**:
  - 不修改 `consecutive_errors` 的增减逻辑
  - 不修改冷却阶梯计算（30s/60s/300s）
  - 不修改 "所有 key 都在 cooldown" 的 fallback 路径（第 56~59 行）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 单文件核心改动（删除一行 sort + 更新 docstring）+ 测试
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 2
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `engine/safety/key_pool.py:40-62` — `acquire_key()` 完整方法，第 61 行是需要删除的 sort
  - `engine/safety/key_pool.py:13-23` — 类 docstring，需更新
  - `engine/safety/key_pool.py:64-98` — `report_rate_limited()`，`consecutive_errors` 用于冷却阶梯计算，不要改
  - `engine/safety/key_pool.py:110-130` — `report_success()`，重置逻辑，不要改

  **Test References**:
  - `tests/test_fallback_truncation.py:286-287` — 现有测试手动 cooldown key 来测试切换，修复后应仍通过

  **WHY Each Reference Matters**:
  - `key_pool.py:61` 的 sort 是 bug 根因——删除它就修复了排序
  - 类 docstring 当前描述与修复后的行为不一致，必须同步更新
  - 冷却阶梯计算使用 `consecutive_errors`，修复后这个字段仍有意义（只是不参与排序）

  **Acceptance Criteria**:

  - [ ] `acquire_key()` 中不再有 `candidates.sort(...)` 调用
  - [ ] docstring 准确描述新行为（insertion order 优先）
  - [ ] Test A/B/C/D 全部通过
  - [ ] `uv run pytest tests/test_fallback_truncation.py -v` 通过（回归）

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Primary with errors=1 recovered after cooldown, fallback has errors=0
    Tool: Bash (uv run pytest)
    Preconditions: Test creates pool with primary + fallback, reports rate_limited on primary
    Steps:
      1. Create APIKeyPool(["primary/model", "fallback/model"])
      2. Call report_rate_limited("primary/model", retry_after_ms=10)  # near-instant cooldown
      3. time.sleep(0.02)  # wait for cooldown to expire
      4. Call acquire_key()
      5. Assert result == "primary/model"
    Expected Result: primary/model is returned despite having consecutive_errors=1
    Failure Indicators: Returns "fallback/model" instead
    Evidence: .sisyphus/evidence/task-1-primary-recovery.txt

  Scenario: All keys in cooldown returns soonest-to-expire
    Tool: Bash (uv run pytest)
    Preconditions: All keys in cooldown with different expiry times
    Steps:
      1. Create pool with 2 keys
      2. Put both in cooldown (first expires sooner)
      3. Call acquire_key()
      4. Assert returns the sooner-to-expire key
    Expected Result: Soonest-to-expire key returned (existing behavior preserved)
    Failure Indicators: Returns wrong key or raises exception
    Evidence: .sisyphus/evidence/task-1-all-cooldown.txt

  Scenario: Regression - existing tests still pass
    Tool: Bash
    Steps:
      1. Run: uv run pytest tests/test_fallback_truncation.py -v
    Expected Result: All tests pass
    Failure Indicators: Any test failure
    Evidence: .sisyphus/evidence/task-1-regression.txt
  ```

  **Commit**: YES
  - Message: `fix(safety): fix key pool sorting to preserve primary priority`
  - Files: `engine/safety/key_pool.py`, `tests/test_key_pool_sorting.py`
  - Pre-commit: `uv run pytest`

- [ ] 2. 移除 `ProviderHealth.pace_level` 死字段

  **What to do**:
  - 在 `engine/providers/provider_models.py` 中：
    - 删除 `ProviderHealth` dataclass 的 `pace_level: PaceLevel = field(default=PaceLevel.HEALTHY)` 字段（第 97 行）
  - 在 `engine/safety/key_pool.py` 中：
    - 修改 `get_cooldown_status()` 方法（第 140~151 行），移除 `pace_level=h.pace_level` 参数
  - 检查是否需要清理 `__init__.py` 导出或其他引用
  - 更新 `docs/codebase-structure.md` 中相关描述
  - 运行 `uv run pytest` 确认所有测试通过

  **Must NOT do**:
  - 不修改 `PaceLevel` 枚举本身（它仍被 `SlidingWindowRateLimiter` 使用）
  - 不修改 `rate_limit.py` 中对 `PaceLevel` 的任何使用

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 删除两个位置的死字段引用
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (after Task 1)
  - **Blocks**: None
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `engine/providers/provider_models.py:89-98` — `ProviderHealth` dataclass，第 97 行是要删除的字段
  - `engine/safety/key_pool.py:140-151` — `get_cooldown_status()`，第 148 行有 `pace_level=h.pace_level` 要删除
  - `engine/providers/provider_models.py:32-37` — `PaceLevel` 枚举，不要删除（rate_limit.py 在用）

  **Test References**:
  - 无现有测试覆盖 `get_cooldown_status()` 或 `ProviderHealth` 构造

  **WHY Each Reference Matters**:
  - `provider_models.py:97` 是死字段的定义处
  - `key_pool.py:148` 是唯一读取该字段的地方（复制到新对象），删除后字段彻底无引用
  - `PaceLevel` 枚举仍被 `SlidingWindowRateLimiter` 使用，不能删

  **Acceptance Criteria**:

  - [ ] `ProviderHealth` 不再有 `pace_level` 字段
  - [ ] `get_cooldown_status()` 不再传递 `pace_level` 参数
  - [ ] `uv run pytest` 全部通过
  - [ ] `docs/codebase-structure.md` 已更新

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: ProviderHealth no longer has pace_level
    Tool: Bash (uv run pytest or python -c)
    Steps:
      1. Run: python -c "from engine.providers.provider_models import ProviderHealth; h = ProviderHealth(profile_name='test'); assert not hasattr(h, 'pace_level')"
    Expected Result: No AttributeError, assertion passes
    Failure Indicators: AttributeError or assertion failure
    Evidence: .sisyphus/evidence/task-2-no-pace-level.txt

  Scenario: Full regression
    Tool: Bash
    Steps:
      1. Run: uv run pytest -v
    Expected Result: All tests pass, 0 failures
    Failure Indicators: Any test failure
    Evidence: .sisyphus/evidence/task-2-regression.txt
  ```

  **Commit**: YES
  - Message: `cleanup(models): remove unused ProviderHealth.pace_level field`
  - Files: `engine/providers/provider_models.py`, `engine/safety/key_pool.py`, `docs/codebase-structure.md`
  - Pre-commit: `uv run pytest`

---

## Final Verification Wave

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan. For each "Must Have": verify implementation exists. For each "Must NOT Have": search for forbidden patterns. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `uv run pytest`. Review changed files for: `as any`, empty catches, unused imports, commented-out code.
  Output: `Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Run test suite, verify sorting behavior with a constructed scenario. Save evidence.
  Output: `Scenarios [N/N pass] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  Verify everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance.
  Output: `Tasks [N/N compliant] | VERDICT`

---

## Commit Strategy

- **1**: `fix(safety): fix key pool sorting to preserve primary priority` - engine/safety/key_pool.py, tests/
- **2**: `cleanup(models): remove unused ProviderHealth.pace_level field` - engine/providers/provider_models.py, engine/safety/key_pool.py

---

## Success Criteria

### Verification Commands
```bash
uv run pytest tests/test_key_pool_sorting.py -v   # Expected: all pass
uv run pytest -v                                     # Expected: all pass
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass
- [ ] docs/codebase-structure.md updated
