"""Path security guard — deny-list based file access control."""

from fnmatch import fnmatch
from pathlib import Path
from typing import List, Optional

# Default patterns for sensitive files that should never be read by agents
DEFAULT_DENIED_PATTERNS: list[str] = [
    "**/.env",
    "**/.env.*",
    "**/.git/**",
    "**/credentials*",
    "**/id_rsa*",
    "**/id_ed25519*",
    "**/id_ecdsa*",
    "**/.ssh/**",
    "**/shadow",
    "**/passwd",
]


class PathGuard:
    """Check file paths against a deny-list of glob patterns.

    Usage:
        guard = PathGuard(denied_patterns=["**/.env"])
        if not guard.is_path_allowed("/home/user/project/.env"):
            reason = guard.check_path("/home/user/project/.env")
            # reason = "Path matches denied pattern: **/.env"
    """

    def __init__(self, denied_patterns: Optional[List[str]] = None):
        self._denied = denied_patterns if denied_patterns is not None else DEFAULT_DENIED_PATTERNS

    def is_path_allowed(self, path: str) -> bool:
        return self.check_path(path) is None

    def check_path(self, path: str) -> Optional[str]:
        try:
            normalized = str(Path(path).resolve())
        except (OSError, ValueError):
            normalized = path

        for pattern in self._denied:
            if fnmatch(normalized, pattern) or fnmatch(path, pattern):
                return f"Path matches denied pattern: {pattern}"
        return None
