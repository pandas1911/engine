import asyncio

import pytest

from engine.safety.rate_limit import SlidingWindowRateLimiter


class TestSchedulerWake:
    @pytest.mark.asyncio
    async def test_release_wakes_blocked_acquire(self):
        limiter = SlidingWindowRateLimiter(rpm_limit=10, tpm_limit=100)

        rid1 = await limiter.acquire(100)
        assert rid1 > 0

        task = asyncio.create_task(limiter.acquire(50))
        await asyncio.sleep(0.1)
        assert not task.done()

        await limiter.release_reserved(rid1)

        result = await asyncio.wait_for(task, timeout=1.0)
        assert result > 0

    @pytest.mark.asyncio
    async def test_record_usage_wakes_on_shrink(self):
        limiter = SlidingWindowRateLimiter(rpm_limit=10, tpm_limit=100)

        rid1 = await limiter.acquire(100)
        assert rid1 > 0

        task = asyncio.create_task(limiter.acquire(50))
        await asyncio.sleep(0.1)
        assert not task.done()

        await limiter.record_usage(30, reservation_id=rid1)

        result = await asyncio.wait_for(task, timeout=1.0)
        assert result > 0

    @pytest.mark.asyncio
    async def test_release_no_waiters_harmless(self):
        limiter = SlidingWindowRateLimiter(rpm_limit=10, tpm_limit=100)

        rid1 = await limiter.acquire(50)
        assert rid1 > 0

        await limiter.release_reserved(rid1)

        rid2 = await asyncio.wait_for(limiter.acquire(50), timeout=1.0)
        assert rid2 > 0

    @pytest.mark.asyncio
    async def test_acquire_immediate_when_capacity(self):
        limiter = SlidingWindowRateLimiter(rpm_limit=10, tpm_limit=100)

        result = await asyncio.wait_for(limiter.acquire(50), timeout=1.0)
        assert result > 0

    @pytest.mark.asyncio
    async def test_multiple_waiters_fifo_wake(self):
        limiter = SlidingWindowRateLimiter(rpm_limit=10, tpm_limit=100)

        rid1 = await limiter.acquire(100)
        assert rid1 > 0

        waiter_a = asyncio.create_task(limiter.acquire(10))
        waiter_b = asyncio.create_task(limiter.acquire(95))
        waiter_c = asyncio.create_task(limiter.acquire(10))

        await asyncio.sleep(0.1)
        assert not waiter_a.done()
        assert not waiter_b.done()
        assert not waiter_c.done()

        await limiter.release_reserved(rid1)

        result_a = await asyncio.wait_for(waiter_a, timeout=1.0)
        assert result_a > 0

        await asyncio.sleep(0.1)
        assert not waiter_b.done()
        assert not waiter_c.done()

        await limiter.release_reserved(result_a)
        result_b = await asyncio.wait_for(waiter_b, timeout=1.0)
        assert result_b > 0
