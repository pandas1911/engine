"""Timezone-aware time formatting utilities for the agent framework."""

import logging
import re
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger(__name__)


class TimeProvider:
    """Provides timezone-aware time formatting utilities."""

    def __init__(self, timezone_override: Optional[str] = None) -> None:
        self._timezone_override = timezone_override
        self._timezone_str: Optional[str] = None

    def resolve_timezone(self) -> str:
        """Resolve the timezone string using override, system tz, or UTC fallback.

        Resolution chain:
        1. Use timezone_override if provided and valid.
        2. Fall back to system local timezone.
        3. Ultimate fallback: "UTC".

        The result is cached after the first call.
        """
        if self._timezone_str is not None:
            return self._timezone_str

        # 1. Try the explicit override
        if self._timezone_override is not None:
            try:
                ZoneInfo(self._timezone_override)
                self._timezone_str = self._timezone_override
                return self._timezone_str
            except ZoneInfoNotFoundError:
                logger.warning(
                    "Invalid timezone override %r, falling back to system timezone.",
                    self._timezone_override,
                )

        # 2. Try the system local timezone
        try:
            system_tz = datetime.now().astimezone().tzinfo
            if system_tz is not None:
                tz_key = getattr(system_tz, "key", None)
                if tz_key:
                    self._timezone_str = tz_key
                    return self._timezone_str
        except Exception:
            logger.warning("Could not detect system timezone, falling back to UTC.")

        # 3. Ultimate fallback
        self._timezone_str = "UTC"
        return self._timezone_str

    def format_system_env_block(self) -> str:
        """Return a formatted environment block with current date and timezone.

        Example::

            <env>
              Today's date: Thu Apr 23 2026
              Time zone: Asia/Shanghai
            </env>
        """
        tz_name = self.resolve_timezone()
        tz_obj = ZoneInfo(tz_name)
        date_str = datetime.now(tz=tz_obj).strftime("%a %b %d %Y")
        return (
            f"<env>\n"
            f"  Today's date: {date_str}\n"
            f"  Time zone: {tz_name}\n"
            f"</env>"
        )

    def format_message_timestamp(self, now: Optional[datetime] = None) -> str:
        """Return a formatted timestamp string like ``[Wed 2026-04-23 14:30 CST]``.

        Parameters
        ----------
        now:
            Optional timezone-aware datetime. If *None*, the current time in the
            resolved timezone is used.
        """
        if now is None:
            tz_name = self.resolve_timezone()
            now = datetime.now(tz=ZoneInfo(tz_name))
        weekday = now.strftime("%a")
        date = now.strftime("%Y-%m-%d")
        time_part = now.strftime("%H:%M")
        tz_abbr = now.tzname() or now.strftime("%Z") or "UTC"
        return f"[{weekday} {date} {time_part} {tz_abbr}]"

    def inject_timestamp(self, message: str, now: Optional[datetime] = None) -> str:
        """Prepend a timestamp to *message* unless one is already present.

        If the message already starts with a bracket-prefixed timestamp
        (matched by ``^\\[.*?\\d{4}-\\d{2}-\\d{2}.*?\\]``), it is returned
        unchanged to avoid double-injection.
        """
        if re.match(r"^\[.*?\d{4}-\d{2}-\d{2}.*?\]", message):
            return message
        ts = self.format_message_timestamp(now)
        return f"{ts} {message}"
