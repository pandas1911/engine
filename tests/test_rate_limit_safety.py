"""Unit tests for Layer 1 deadlock prevention: estimated_tokens cap in acquire()."""

import asyncio

import pytest

from engine.safety.rate_limit import SlidingWindowRateLimiter
from engine.safety.token_estimator import EmaTokenEstimator


@pytest.mark.asyncio
async def test_acquire_caps_estimated_tokens_above_tpm_limit():
    """Huge estimated_tokens is capped so acquire returns immediately."""
    limiter = SlidingWindowRateLimiter(rpm_limit=100, tpm_limit=20, window_seconds=0.5)

    rid = await asyncio.wait_for(
        limiter.acquire(estimated_tokens=999999), timeout=5.0
    )

    assert rid > 0
    assert limiter._current_tpm() <= 20


@pytest.mark.asyncio
async def test_acquire_caps_at_exact_tpm_limit():
    """estimated_tokens == tpm_limit is capped to tpm_limit - 1."""
    limiter = SlidingWindowRateLimiter(rpm_limit=100, tpm_limit=20, window_seconds=0.5)

    rid = await limiter.acquire(estimated_tokens=20)

    assert rid > 0
    assert limiter._current_tpm() == 19


@pytest.mark.asyncio
async def test_tpm_disabled_no_cap():
    """When tpm_limit=0, cap is skipped but acquire still completes without deadlock."""
    limiter = SlidingWindowRateLimiter(rpm_limit=10, tpm_limit=0, window_seconds=0.5)

    rid = await asyncio.wait_for(
        limiter.acquire(estimated_tokens=999999), timeout=5.0
    )

    assert rid > 0


@pytest.mark.asyncio
async def test_normal_flow_unaffected():
    """Normal acquire → record_usage → acquire cycle works correctly."""
    limiter = SlidingWindowRateLimiter(rpm_limit=100, tpm_limit=100, window_seconds=0.5)

    rid = await limiter.acquire(estimated_tokens=10)
    assert rid > 0

    await limiter.record_usage(8, reservation_id=rid)

    rid2 = await limiter.acquire(estimated_tokens=10)
    assert rid2 > 0


# ---------------------------------------------------------------------------
# Layer 2: deadlock detection + acquire timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scheduler_force_release_on_deadlock():
    """Scheduler wakes blocked waiter when window entries expire."""
    limiter = SlidingWindowRateLimiter(rpm_limit=10, tpm_limit=5, window_seconds=0.3)

    await limiter.acquire(estimated_tokens=5)  # fills TPM (capped to 4)

    task = asyncio.create_task(limiter.acquire(estimated_tokens=5))
    await asyncio.sleep(0.1)

    rid = await asyncio.wait_for(task, timeout=5.0)
    assert rid > 0


@pytest.mark.asyncio
async def test_acquire_timeout_raises():
    """Blocked acquire raises TimeoutError when external timeout fires."""
    limiter = SlidingWindowRateLimiter(rpm_limit=10, tpm_limit=5, window_seconds=0.5)

    await limiter.acquire(estimated_tokens=5)  # fills TPM (capped to 4)

    task = asyncio.create_task(limiter.acquire(estimated_tokens=5))
    await asyncio.sleep(0.05)  # let task enter slow path

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(task, timeout=0.1)


@pytest.mark.asyncio
async def test_fifo_ordering_preserved():
    """Multiple waiters are all released when window expires."""
    limiter = SlidingWindowRateLimiter(rpm_limit=10, tpm_limit=20, window_seconds=0.3)

    await limiter.acquire(estimated_tokens=20)  # fills TPM (capped to 19)

    t1 = asyncio.create_task(limiter.acquire(estimated_tokens=5))
    t2 = asyncio.create_task(limiter.acquire(estimated_tokens=5))
    t3 = asyncio.create_task(limiter.acquire(estimated_tokens=5))

    await asyncio.sleep(0.4)  # wait for window to expire
    r1, r2, r3 = await asyncio.gather(t1, t2, t3)

    assert r1 > 0
    assert r2 > 0
    assert r3 > 0


@pytest.mark.asyncio
async def test_window_expiry_releases_waiters():
    """Waiter is released when window entries expire and scheduler runs."""
    limiter = SlidingWindowRateLimiter(rpm_limit=10, tpm_limit=10, window_seconds=0.3)

    await limiter.acquire(estimated_tokens=10)  # fills TPM (capped to 9)

    task = asyncio.create_task(limiter.acquire(estimated_tokens=5))

    await asyncio.sleep(0.5)  # wait for window expiry + scheduler cycle
    rid = await asyncio.wait_for(task, timeout=2.0)

    assert rid > 0


# ---------------------------------------------------------------------------
# Layer 3: EmaTokenEstimator
# ---------------------------------------------------------------------------


def test_ema_coefficient_converges():
    """Repeated 3x underestimate feedback moves coefficient toward 1.0."""
    estimator = EmaTokenEstimator(coefficient=3.0)
    for _ in range(5):
        estimator.feedback(1000, 3000)
    assert estimator.coefficient < 3.0


def test_ema_coefficient_bounds():
    """Coefficient is clamped to [1.0, 5.0] after feedback."""
    est_low = EmaTokenEstimator(coefficient=3.0)
    est_low.feedback(1, 10000)  # ideal=0.0003, clamped to 1.0
    assert est_low.coefficient >= 1.0

    est_high = EmaTokenEstimator(coefficient=3.0)
    est_high.feedback(10000, 1)  # ideal=30000, clamped to 5.0
    assert est_high.coefficient <= 5.0


def test_ema_skips_zero_estimated():
    """feedback with estimated_tokens=0 leaves coefficient unchanged."""
    estimator = EmaTokenEstimator(coefficient=3.0)
    estimator.feedback(0, 5000)
    assert estimator.coefficient == 3.0


def test_ema_skips_zero_actual():
    """feedback with actual_tokens=0 leaves coefficient unchanged."""
    estimator = EmaTokenEstimator(coefficient=3.0)
    estimator.feedback(1000, 0)
    assert estimator.coefficient == 3.0


def test_estimator_estimate_returns_int():
    """estimate() returns a positive int."""
    estimator = EmaTokenEstimator()
    result = estimator.estimate([{"role": "user", "content": "hello"}], None)
    assert isinstance(result, int)
    assert result >= 1
