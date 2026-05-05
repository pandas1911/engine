"""Unit tests for SlidingWindowRateLimiter with integrated adaptive pacing.

Validates pacing behavior, pace-level transitions, RPM/TPM limits,
lock separation, and backward compatibility — all in-memory, no mocks.
"""

import asyncio
import time
from collections import deque

import pytest

from engine.safety.rate_limit import SlidingWindowRateLimiter
from engine.providers.provider_models import PaceLevel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _limiter(
    *,
    rpm_limit: float = 100,
    tpm_limit: float = 100_000,
    pacing_enabled: bool = False,
    min_interval_ms: float = 500.0,
    window_seconds: float = 60.0,
) -> SlidingWindowRateLimiter:
    return SlidingWindowRateLimiter(
        rpm_limit=rpm_limit,
        tpm_limit=tpm_limit,
        window_seconds=window_seconds,
        profile_name="test",
        pacing_enabled=pacing_enabled,
        min_interval_ms=min_interval_ms,
    )


def _seed_rpm(lim: SlidingWindowRateLimiter, count: int) -> None:
    for _ in range(count):
        lim._rpm_entries.append((time.monotonic(), 0))


# ---------------------------------------------------------------------------
# 1. Pacing disabled — no delay
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pacing_disabled_no_delay() -> None:
    lim = _limiter(pacing_enabled=False, rpm_limit=100, tpm_limit=100_000)

    start = time.monotonic()
    await lim.acquire()
    await lim.acquire()
    elapsed_ms = (time.monotonic() - start) * 1000.0

    assert elapsed_ms < 50, f"Expected < 50 ms, got {elapsed_ms:.1f} ms"


# ---------------------------------------------------------------------------
# 2. Pacing min_interval enforced
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pacing_min_interval_enforced() -> None:
    lim = _limiter(
        pacing_enabled=True,
        min_interval_ms=500,
        rpm_limit=100,
        tpm_limit=100_000,
    )

    await lim.acquire()

    start = time.monotonic()
    await lim.acquire()
    elapsed_ms = (time.monotonic() - start) * 1000.0

    assert elapsed_ms >= 450, f"Expected >= 450 ms, got {elapsed_ms:.1f} ms"


# ---------------------------------------------------------------------------
# 3. Pacing concurrent stagger
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pacing_concurrent_stagger() -> None:
    lim = _limiter(
        pacing_enabled=True,
        min_interval_ms=200,
        rpm_limit=100,
        tpm_limit=100_000,
    )

    timestamps: list[float] = []

    async def stamped_acquire() -> None:
        await lim.acquire()
        timestamps.append(time.monotonic())

    await asyncio.gather(
        stamped_acquire(),
        stamped_acquire(),
        stamped_acquire(),
    )

    timestamps.sort()
    assert len(timestamps) == 3

    gap1 = (timestamps[1] - timestamps[0]) * 1000.0
    gap2 = (timestamps[2] - timestamps[1]) * 1000.0

    assert gap1 >= 100, f"Gap 1 too short: {gap1:.0f} ms"
    assert gap2 >= 100, f"Gap 2 too short: {gap2:.0f} ms"


# ---------------------------------------------------------------------------
# 4. Pace level — HEALTHY
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pace_level_healthy() -> None:
    lim = _limiter(pacing_enabled=True, rpm_limit=10, tpm_limit=100_000)

    _seed_rpm(lim, 3)
    await lim.record_usage(0)

    assert lim._pace_level is PaceLevel.HEALTHY
    assert lim._get_recommended_delay() == 0.0


# ---------------------------------------------------------------------------
# 5. Pace level — PRESSING
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pace_level_pressing() -> None:
    lim = _limiter(pacing_enabled=True, rpm_limit=10, tpm_limit=100_000)

    _seed_rpm(lim, 6)
    await lim.record_usage(0)

    assert lim._pace_level is PaceLevel.PRESSING
    assert lim._get_recommended_delay() == 200.0


# ---------------------------------------------------------------------------
# 6. Pace level — CRITICAL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pace_level_critical() -> None:
    lim = _limiter(pacing_enabled=True, rpm_limit=10, tpm_limit=100_000)

    _seed_rpm(lim, 9)
    await lim.record_usage(0)

    assert lim._pace_level is PaceLevel.CRITICAL
    assert lim._get_recommended_delay() == 1000.0


# ---------------------------------------------------------------------------
# 7. Pace level transitions on record_usage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pace_level_updates_on_record_usage() -> None:
    lim = _limiter(pacing_enabled=True, rpm_limit=10, tpm_limit=100_000)

    _seed_rpm(lim, 3)
    await lim.record_usage(0)
    assert lim._pace_level is PaceLevel.HEALTHY

    _seed_rpm(lim, 3)
    await lim.record_usage(0)
    assert lim._pace_level is PaceLevel.PRESSING

    _seed_rpm(lim, 3)
    await lim.record_usage(0)
    assert lim._pace_level is PaceLevel.CRITICAL


# ---------------------------------------------------------------------------
# 8. RPM limit blocks excess requests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rpm_limit_with_pacing() -> None:
    lim = _limiter(pacing_enabled=False, rpm_limit=3, tpm_limit=100_000)

    await lim.acquire()
    await lim.acquire()
    await lim.acquire()

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(lim.acquire(), timeout=0.2)


# ---------------------------------------------------------------------------
# 9. TPM limit blocks over-budget requests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tpm_limit_with_pacing() -> None:
    lim = _limiter(
        pacing_enabled=False,
        rpm_limit=100,
        tpm_limit=100,
        window_seconds=0.3,
    )

    rid1 = await lim.acquire(estimated_tokens=80)
    assert rid1 > 0

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(lim.acquire(estimated_tokens=30), timeout=0.2)


# ---------------------------------------------------------------------------
# 10. Pacing does not hold main lock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pacing_does_not_hold_main_lock() -> None:
    lim = _limiter(
        pacing_enabled=True,
        min_interval_ms=2000,
        rpm_limit=100,
        tpm_limit=100_000,
    )

    async def slow_acquire() -> None:
        await lim.acquire()

    task = asyncio.create_task(slow_acquire())
    await asyncio.sleep(0)

    start = time.monotonic()
    await lim.record_usage(0)
    elapsed_ms = (time.monotonic() - start) * 1000.0

    assert elapsed_ms < 200, f"record_usage blocked by pacing sleep: {elapsed_ms:.0f} ms"

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# 11. Effective min_interval derived from RPM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_effective_min_interval_from_rpm() -> None:
    lim = _limiter(
        pacing_enabled=True,
        rpm_limit=10,
        min_interval_ms=100,
    )

    assert lim._effective_min_interval_ms == 6000.0


# ---------------------------------------------------------------------------
# 12. Existing limiter behavior unchanged (pacing disabled)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_existing_limiter_behavior_unchanged() -> None:
    lim = _limiter(pacing_enabled=False, rpm_limit=5, tpm_limit=1000)

    start = time.monotonic()

    rid1 = await lim.acquire(0)
    assert rid1 == 0

    rid2 = await lim.acquire(estimated_tokens=50)
    assert rid2 > 0

    await lim.record_usage(45, rid2)
    await lim.release_reserved(rid2)

    rid3 = await lim.acquire(estimated_tokens=30)
    assert rid3 > 0
    await lim.release_reserved(rid3)

    total_ms = (time.monotonic() - start) * 1000.0
    assert total_ms < 200, f"Unexpected delay: {total_ms:.0f} ms"
