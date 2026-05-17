"""Security analysis for bash command execution.

Implements layered security:
1. Dangerous command blocklist (hard block)
2. Safe command allowlist (fast-path)
3. Unknown command warning
4. External/dangerous path detection
5. Dangerous redirect detection
6. Environment variable sanitization
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional

try:
    from engine.tools.builtin._bash.ast_parser import ParsedCommand
except ImportError:
    from dataclasses import dataclass, field
    from typing import List, Optional

    @dataclass
    class ParsedCommand:  # type: ignore[no-redef]
        command_names: List[str] = field(default_factory=list)
        file_paths: List[str] = field(default_factory=list)
        redirects: list = field(default_factory=list)
        has_pipes: bool = False
        has_dynamic_parts: bool = False
        raw_command: str = ""

logger = logging.getLogger(__name__)


@dataclass
class SecurityResult:
    """Result of security analysis for a command."""
    allowed: bool
    reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


# Blocked Commands (hard block -- never execute)
BLOCKED_COMMANDS: FrozenSet[str] = frozenset({
    "mkfs", "format", "dd",
    "shutdown", "reboot", "halt", "poweroff", "init",
    "forkbomb",
})

BLOCKED_PATTERNS: List[tuple[str, str]] = [
    (r"rm\s+-rf\s+/", "Refusing to recursively force-delete root directory"),
    (r"rm\s+-r\s+/", "Refusing to recursively delete root directory"),
    (r"rm\s+-rf\s+~", "Refusing to recursively force-delete home directory"),
    (r"chmod\s+777\s+/", "Refusing to make root world-writable"),
    (r"chmod\s+-R\s+777", "Refusing to recursively make world-writable"),
    (r":\(\)\{\s*:\|\:&\s*\}", "Fork bomb pattern detected"),
]


# Safe Commands (allowlist -- execute without warning)
SAFE_COMMANDS: FrozenSet[str] = frozenset({
    "ls", "cat", "head", "tail", "less", "more", "wc", "stat",
    "file", "du", "df", "tree", "find", "which", "type",
    "grep", "egrep", "fgrep", "rg", "ag", "ack",
    "sort", "uniq", "cut", "tr", "paste", "column", "tee",
    "diff", "comm", "jq", "yq",
    "git", "svn", "hg",
    "npm", "yarn", "pnpm", "bun", "npx",
    "pip", "pip3", "uv", "poetry", "conda",
    "cargo", "go", "make", "cmake", "gradle", "mvn",
    "pytest", "jest", "vitest", "mocha", "unittest",
    "node", "python", "python3", "ruby", "perl", "java",
    "bash", "sh", "zsh",
    "docker", "kubectl", "terraform", "helm",
    "curl", "wget", "ping", "host", "dig", "nslookup",
    "ssh", "scp", "rsync",
    "echo", "printf", "date", "uname", "hostname",
    "whoami", "id", "env", "printenv", "export",
    "ps", "top", "htop", "uptime",
    "mkdir", "touch", "ln", "cp", "mv",
    "tar", "gzip", "gunzip", "zip", "unzip",
    "xargs", "sed", "awk",
})


# Dangerous System Paths (never write to)
DANGEROUS_PATHS: FrozenSet[str] = frozenset({
    "/etc", "/sys", "/proc", "/boot", "/root",
    "/var/lib", "/usr", "/sbin", "/bin",
    "/etc/passwd", "/etc/shadow", "/etc/sudoers",
})


# Environment Variable Sanitization
BLOCKED_ENV_KEYS: FrozenSet[str] = frozenset({
    # Code injection vectors
    "NODE_OPTIONS", "NODE_PATH", "PYTHONPATH", "PYTHONHOME",
    "PERL5LIB", "PERL5OPT", "RUBYLIB", "RUBYOPT",
    "BASH_ENV", "ENV",
    # Dynamic library injection
    "LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH",
    # Git manipulation
    "GIT_DIR", "GIT_WORK_TREE", "GIT_EXEC_PATH", "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY", "GIT_NAMESPACE", "GIT_TEMPLATE_DIR",
    "GIT_SSL_NO_VERIFY", "GIT_SEQUENCE_EDITOR", "GIT_EXTERNAL_DIFF",
    "GIT_EDITOR", "GIT_HOOK_PATH",
    # Compiler/build injection
    "CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS", "LIBRARY_PATH",
    "CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER", "CMAKE_TOOLCHAIN_FILE",
    "RUSTFLAGS", "RUSTC_WRAPPER", "CARGO_BUILD_RUSTC", "GOFLAGS",
    # Shell manipulation
    "SHELL", "SHELLOPTS", "IFS", "PS4",
    # TLS/cert bypass
    "NODE_TLS_REJECT_UNAUTHORIZED", "SSL_CERT_FILE", "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
    # Java / Python / Others
    "JAVA_OPTS", "JAVA_TOOL_OPTIONS", "_JAVA_OPTIONS", "JDK_JAVA_OPTIONS",
    "PYTHONBREAKPOINT", "PYTHONSTARTUP",
    "DOTNET_STARTUP_HOOKS", "LUA_INIT", "EMACSLOADPATH",
    "MAVEN_OPTS", "GRADLE_OPTS", "SBT_OPTS", "ANT_OPTS",
    "CONFIG_SITE", "CONFIG_SHELL",
})

BLOCKED_ENV_PREFIXES: FrozenSet[str] = frozenset({
    "DYLD_", "LD_", "BASH_FUNC_",
    "GIT_CONFIG_", "NPM_CONFIG_", "CARGO_REGISTRIES_",
})


class SecurityChecker:
    """Check commands against security policies before execution."""

    def check_command(self, parsed: ParsedCommand, project_root: Optional[str] = None) -> SecurityResult:
        warnings: List[str] = []
        block_reason = self._check_blocklist(parsed)
        if block_reason:
            return SecurityResult(allowed=False, reason=block_reason)
        redirect_warning = self._check_redirects(parsed)
        if redirect_warning:
            return SecurityResult(allowed=False, reason=redirect_warning)
        path_warning = self._check_paths(parsed, project_root or "")
        if path_warning:
            warnings.append(path_warning)
        if self._all_commands_known_safe(parsed):
            return SecurityResult(allowed=True, warnings=warnings)
        unknown = self._get_unknown_commands(parsed)
        if unknown:
            warnings.append(f"Unknown command(s) not in allowlist: {', '.join(unknown)}.")
        return SecurityResult(allowed=True, warnings=warnings)

    def _check_blocklist(self, parsed: ParsedCommand) -> Optional[str]:
        for cmd_name in parsed.command_names:
            if cmd_name in BLOCKED_COMMANDS:
                return f"Command '{cmd_name}' is blocked for safety."
        for pattern, reason in BLOCKED_PATTERNS:
            if re.search(pattern, parsed.raw_command):
                return reason
        return None

    def _check_redirects(self, parsed: ParsedCommand) -> Optional[str]:
        for redirect in parsed.redirects:
            for dangerous in DANGEROUS_PATHS:
                if redirect.target_path.startswith(dangerous):
                    return f"Redirect to dangerous path blocked: {redirect.target_path}"
        return None

    def _check_paths(self, parsed: ParsedCommand, project_root: str) -> Optional[str]:
        if not project_root or not parsed.file_paths:
            return None
        external = [p for p in parsed.file_paths if not p.startswith(project_root)]
        if external:
            return f"Command accesses path(s) outside project: {', '.join(external[:3])}"
        return None

    def _all_commands_known_safe(self, parsed: ParsedCommand) -> bool:
        return bool(parsed.command_names) and all(c in SAFE_COMMANDS for c in parsed.command_names)

    @staticmethod
    def _get_unknown_commands(parsed: ParsedCommand) -> List[str]:
        return [c for c in parsed.command_names if c not in SAFE_COMMANDS]


def sanitize_env(user_env: Dict[str, str], base_env: Dict[str, str]) -> Dict[str, str]:
    """Merge user env into base env with sanitization. Never allows PATH override."""
    result = dict(base_env)
    for key, value in user_env.items():
        if key.upper() == "PATH":
            logger.warning("Blocked PATH override from agent env")
            continue
        if key in BLOCKED_ENV_KEYS:
            logger.warning("Blocked dangerous env key: %s", key)
            continue
        if any(key.upper().startswith(p) for p in BLOCKED_ENV_PREFIXES):
            logger.warning("Blocked env key by prefix: %s", key)
            continue
        result[key] = value
    return result
