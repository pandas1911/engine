"""EMA-based token estimator and result truncation utilities."""

from typing import Dict, List, Optional


class EmaTokenEstimator:
    """EMA-based token estimator that learns chars-to-tokens ratio over time.

    Instead of a fixed chars//3 formula, the coefficient adapts via exponential
    moving average (EMA) feedback after each successful LLM call.
    """

    def __init__(
        self,
        coefficient: float = 3.0,
        alpha: float = 0.2,
        min_coefficient: float = 1.0,
        max_coefficient: float = 5.0,
    ):
        self._coefficient = coefficient
        self._alpha = alpha
        self._min = min_coefficient
        self._max = max_coefficient

    def estimate(self, messages: List[Dict], tools: Optional[List[Dict]]) -> int:
        """Estimate token count from messages and tools using current coefficient."""
        total_chars = sum(len(str(m)) for m in messages)
        total_chars += sum(len(str(t)) for t in (tools or []))
        return max(1, int(total_chars / self._coefficient))

    def feedback(self, estimated_tokens: int, actual_tokens: int) -> None:
        """Update coefficient via EMA after observing actual token usage.

        Derivation:
          estimated = total_chars / coefficient
          We want: estimated ~ actual_tokens
          So: coefficient ~ total_chars / actual_tokens
          Since total_chars = estimated_tokens * current_coefficient:
          ideal_coefficient = estimated_tokens * current_coefficient / actual_tokens
        """
        if estimated_tokens <= 0 or actual_tokens <= 0:
            return
        ideal_coefficient = (estimated_tokens * self._coefficient) / actual_tokens
        clamped = max(self._min, min(self._max, ideal_coefficient))
        self._coefficient = (
            self._alpha * clamped
            + (1 - self._alpha) * self._coefficient
        )

    @property
    def coefficient(self) -> float:
        """Current coefficient value (for testing/monitoring)."""
        return self._coefficient


class ResultTruncator:
    """Truncates results to prevent unbounded result sizes."""

    @staticmethod
    def truncate(result: str, max_length: int) -> str:
        """Truncate a result string if it exceeds max_length.

        Args:
            result: The result string to truncate
            max_length: Maximum allowed length

        Returns:
            The original string if within limits, or truncated string with suffix
            that includes the original character count.
        """
        original_length = len(result)
        if original_length <= max_length:
            return result

        suffix = "[truncated, original: {} chars]".format(original_length)
        available_length = max_length - len(suffix)
        if available_length <= 0:
            return result[:max_length]

        return result[:available_length] + suffix
