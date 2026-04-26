"""Retry engine with exponential backoff extracted from engine.safety.

Provides error classification and retry logic with configurable jitter.
"""

import asyncio
import random
import re
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


class RetryEngine:
    """Error classification and retry logic with exponential backoff and jitter.

    Handles classification of errors into RATE_LIMITED, RETRYABLE, and
    NON_RETRYABLE categories, extracts retry-after hints from error messages,
    computes delays with exponential backoff and configurable jitter, and
    executes async functions with automatic retry.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter_mode: str = "symmetric",
    ):
        if max_attempts < 1:
            raise ValueError(
                "max_attempts must be >= 1, got {}".format(max_attempts)
            )
        if base_delay <= 0:
            raise ValueError(
                "base_delay must be > 0, got {}".format(base_delay)
            )
        if max_delay <= 0:
            raise ValueError(
                "max_delay must be > 0, got {}".format(max_delay)
            )
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter_mode = jitter_mode

    def classify_error(self, error: Exception) -> Any:
        """Classify an exception into an ErrorClass.

        Checks error string for common HTTP status codes and rate-limit
        patterns. Order matters: RATE_LIMITED first, then NON_RETRYABLE,
        then default RETRYABLE.
        """
        from engine.providers.provider_models import ErrorClass

        msg = str(error).lower()

        rate_limited_patterns = [
            "429",
            "rate limit",
            "too many requests",
            "quota exceeded",
            "resource_exhausted",
        ]
        if any(p in msg for p in rate_limited_patterns):
            return ErrorClass.RATE_LIMITED

        non_retryable_patterns = [
            "401",
            "403",
            "authentication",
            "invalid api key",
            "unauthorized",
        ]
        if any(p in msg for p in non_retryable_patterns):
            return ErrorClass.NON_RETRYABLE

        return ErrorClass.RETRYABLE

    def extract_retry_after(self, error: Exception) -> Optional[float]:
        """Extract retry-after value from error message in milliseconds.

        Checks for common patterns like 'retry-after: 30', 'retry after 5',
        and 'try again in X seconds'. Returns value in milliseconds, or
        None if no pattern is found.
        """
        msg = str(error)

        # Pattern: retry-after: 30 (with optional unit)
        match = re.search(
            r"retry-after[:\s]+(\d+(?:\.\d+)?)\s*(ms|milliseconds?)?",
            msg,
            re.IGNORECASE,
        )
        if match:
            value = float(match.group(1))
            if match.group(2):
                return value
            return value * 1000.0

        # Pattern: retry after 5 (without colon)
        match = re.search(
            r"retry after[:\s]+(\d+(?:\.\d+)?)\s*(ms|milliseconds?)?",
            msg,
            re.IGNORECASE,
        )
        if match:
            value = float(match.group(1))
            if match.group(2):
                return value
            return value * 1000.0

        # Pattern: try again in X seconds
        match = re.search(
            r"try again in[:\s]+(\d+(?:\.\d+)?)\s*(ms|milliseconds?|s|seconds?)?",
            msg,
            re.IGNORECASE,
        )
        if match:
            value = float(match.group(1))
            unit = match.group(2) or ""
            if unit.lower().startswith("ms"):
                return value
            return value * 1000.0

        return None

    def compute_delay(
        self, attempt: int, retry_after_ms: Optional[float] = None
    ) -> float:
        """Compute delay for a retry attempt with exponential backoff and jitter.

        Args:
            attempt: Current attempt number (1-based).
            retry_after_ms: Optional retry-after hint from provider in ms.

        Returns:
            Delay in seconds to wait before next attempt.
        """
        base = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)

        if retry_after_ms is not None:
            base = max(base, retry_after_ms / 1000.0)

        if retry_after_ms is not None:
            # Positive jitter only when retry_after is present
            jitter = random.uniform(1.0, 1.5)
        else:
            jitter = random.uniform(0.5, 1.5)

        return base * jitter

    async def execute_with_retry(
        self,
        fn: Callable[..., Any],
        on_retry: Optional[Callable[[int, Exception, float], None]] = None,
    ) -> Any:
        """Execute an async function with retry logic.

        Args:
            fn: Async callable to execute.
            on_retry: Optional callback invoked on each retry with
                (attempt, error, delay_seconds).

        Returns:
            Result from fn().

        Raises:
            LLMProviderError: When all retry attempts are exhausted.
            Exception: When a NON_RETRYABLE error is encountered.
        """
        from engine.logging import get_logger
        from engine.providers.llm_provider import LLMProviderError
        from engine.providers.provider_models import ErrorClass

        logger = get_logger()
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                return await fn()
            except Exception as e:
                last_error = e
                error_class = self.classify_error(e)

                if error_class == ErrorClass.NON_RETRYABLE:
                    raise

                if attempt >= self.max_attempts:
                    break

                retry_after_ms = self.extract_retry_after(e)
                delay = self.compute_delay(attempt, retry_after_ms)

                logger.warning(
                    "RateControl",
                    "Retry attempt | attempt={}/{}, delay={:.2f}s, error_class={}, error='{}'".format(
                        attempt, self.max_attempts, delay,
                        error_class.value, str(e)[:200]
                    ),
                    event_type="retry_attempt",
                    data={
                        "attempt": attempt,
                        "max_attempts": self.max_attempts,
                        "delay_seconds": delay,
                        "error_class": error_class.value,
                        "error_message": str(e)[:500],
                    },
                )

                if on_retry is not None:
                    on_retry(attempt, e, delay)

                await asyncio.sleep(delay)

        raise LLMProviderError(last_error)
