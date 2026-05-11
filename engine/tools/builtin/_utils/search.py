"""Search engine abstraction — ripgrep with Python fallback."""

import json
import re
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class SearchResult:
    """A single search match."""
    file: str
    line_number: int
    content: str


MAX_RESULTS = 100
MAX_LINE_LENGTH = 2000


class SearchEngine(ABC):
    """Abstract base for content and file search."""

    @abstractmethod
    def search_content(
        self,
        pattern: str,
        path: str,
        include: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search file contents for a regex pattern."""
        ...

    @abstractmethod
    def search_files(
        self,
        pattern: str,
        path: str,
    ) -> List[str]:
        """Find files matching a glob pattern."""
        ...


class RipgrepEngine(SearchEngine):
    """Search using ripgrep (rg) binary."""

    def search_content(self, pattern: str, path: str, include: Optional[str] = None) -> List[SearchResult]:
        cmd = ["rg", "--json", "--max-count", str(MAX_RESULTS)]
        if include:
            cmd.extend(["--glob", include])
        cmd.extend([pattern, path])

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []

        results: list[SearchResult] = []
        for line in proc.stdout.strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get("type") == "match":
                    d = data["data"]
                    text = d["lines"].get("text", "")[:MAX_LINE_LENGTH]
                    results.append(SearchResult(
                        file=d["path"]["text"],
                        line_number=d["line_number"],
                        content=text.rstrip("\n"),
                    ))
            except (json.JSONDecodeError, KeyError):
                continue
            if len(results) >= MAX_RESULTS:
                break
        return results

    def search_files(self, pattern: str, path: str) -> List[str]:
        cmd = ["rg", "--files", "--glob", pattern, path]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []
        files = [f for f in proc.stdout.strip().split("\n") if f]
        return files[:MAX_RESULTS]


class PythonEngine(SearchEngine):
    """Pure Python fallback search using pathlib + re."""

    def search_content(self, pattern: str, path: str, include: Optional[str] = None) -> List[SearchResult]:
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}") from e

        results: list[SearchResult] = []
        root = Path(path)
        if root.is_file():
            files = [root]
        else:
            glob_pattern = include or "**/*"
            files = root.glob(glob_pattern)

        for fp in files:
            if not fp.is_file():
                continue
            try:
                with open(fp, "r", errors="replace") as f:
                    for i, line in enumerate(f, 1):
                        if compiled.search(line):
                            results.append(SearchResult(
                                file=str(fp),
                                line_number=i,
                                content=line.rstrip("\n")[:MAX_LINE_LENGTH],
                            ))
                            if len(results) >= MAX_RESULTS:
                                return results
            except (OSError, UnicodeDecodeError):
                continue
        return results

    def search_files(self, pattern: str, path: str) -> List[str]:
        root = Path(path)
        files = [str(f) for f in root.glob(pattern) if f.is_file()]
        return files[:MAX_RESULTS]


def get_search_engine() -> SearchEngine:
    """Return RipgrepEngine if rg is available, else PythonEngine."""
    if shutil.which("rg"):
        return RipgrepEngine()
    return PythonEngine()
