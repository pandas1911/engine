import time

import pytest

from engine.safety.key_pool import APIKeyPool


class TestKeyPoolSorting:
    def test_primary_recovered_after_cooldown_despite_higher_errors(self):
        pool = APIKeyPool(
            names=["primary/model", "fallback/model"],
            cooldown_initial_ms=10,
            cooldown_max_ms=10,
        )
        pool.report_rate_limited("primary/model")
        time.sleep(0.02)
        result = pool.acquire_key()
        assert result == "primary/model"

    def test_all_in_cooldown_returns_soonest_expiry(self):
        pool = APIKeyPool(names=["primary/model", "fallback/model"])
        pool.report_rate_limited("primary/model", retry_after_ms=100)
        pool.report_rate_limited("fallback/model", retry_after_ms=500)
        result = pool.acquire_key()
        assert result == "primary/model"

    def test_report_success_resets_errors_and_cooldown(self):
        pool = APIKeyPool(names=["primary/model"])
        pool.report_rate_limited("primary/model", retry_after_ms=1000)
        assert pool._health["primary/model"].consecutive_errors == 1
        assert pool._health["primary/model"].cooldown_until is not None
        pool.report_success("primary/model")
        assert pool._health["primary/model"].consecutive_errors == 0
        assert pool._health["primary/model"].cooldown_until is None

    def test_single_key_pool_unaffected(self):
        pool = APIKeyPool(names=["only/model"])
        assert pool.acquire_key() == "only/model"
        pool.report_rate_limited("only/model", retry_after_ms=10)
        time.sleep(0.02)
        assert pool.acquire_key() == "only/model"
