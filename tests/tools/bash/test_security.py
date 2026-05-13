"""Tests for engine/tools/builtin/_bash/security.py."""

from __future__ import annotations

import pytest

try:
    from engine.tools.builtin._bash.ast_parser import ParsedCommand, RedirectInfo
except ImportError:
    from dataclasses import dataclass, field
    from typing import List, Optional

    @dataclass
    class RedirectInfo:
        target_path: str
        mode: str

    @dataclass
    class ParsedCommand:
        command_names: List[str] = field(default_factory=list)
        file_paths: List[str] = field(default_factory=list)
        redirects: List[RedirectInfo] = field(default_factory=list)
        has_pipes: bool = False
        has_dynamic_parts: bool = False
        raw_command: str = ""


from engine.tools.builtin._bash.security import (
    BLOCKED_COMMANDS,
    BLOCKED_ENV_KEYS,
    SAFE_COMMANDS,
    SecurityChecker,
    sanitize_env,
)


@pytest.fixture
def checker() -> SecurityChecker:
    return SecurityChecker()


class TestBlocklist:
    def test_blocked_command_mkfs(self, checker: SecurityChecker) -> None:
        parsed = ParsedCommand(command_names=["mkfs"], raw_command="mkfs /dev/sda1")
        result = checker.check_command(parsed)
        assert result.allowed is False
        assert "mkfs" in result.reason

    def test_blocked_command_dd(self, checker: SecurityChecker) -> None:
        parsed = ParsedCommand(command_names=["dd"], raw_command="dd if=/dev/zero of=/dev/sda")
        result = checker.check_command(parsed)
        assert result.allowed is False
        assert "dd" in result.reason

    def test_blocked_command_shutdown(self, checker: SecurityChecker) -> None:
        parsed = ParsedCommand(command_names=["shutdown"], raw_command="shutdown -h now")
        result = checker.check_command(parsed)
        assert result.allowed is False
        assert "shutdown" in result.reason

    def test_blocked_pattern_rm_rf_root(self, checker: SecurityChecker) -> None:
        parsed = ParsedCommand(command_names=["rm"], raw_command="rm -rf /")
        result = checker.check_command(parsed)
        assert result.allowed is False
        assert "root" in result.reason.lower() or "force-delete" in result.reason.lower()

    def test_blocked_pattern_fork_bomb(self, checker: SecurityChecker) -> None:
        parsed = ParsedCommand(command_names=[":"], raw_command=":(){ :|:& };:")
        result = checker.check_command(parsed)
        assert result.allowed is False
        assert "Fork bomb" in result.reason

    def test_all_blocked_commands_are_blocked(self, checker: SecurityChecker) -> None:
        for cmd in BLOCKED_COMMANDS:
            parsed = ParsedCommand(command_names=[cmd], raw_command=cmd)
            result = checker.check_command(parsed)
            assert result.allowed is False, f"Blocked command '{cmd}' was allowed"


class TestAllowlist:
    def test_safe_command_git(self, checker: SecurityChecker) -> None:
        parsed = ParsedCommand(command_names=["git"], raw_command="git status")
        result = checker.check_command(parsed)
        assert result.allowed is True
        assert result.warnings == []

    def test_safe_command_multiple(self, checker: SecurityChecker) -> None:
        parsed = ParsedCommand(command_names=["git", "grep"], raw_command="git log | grep fix")
        result = checker.check_command(parsed)
        assert result.allowed is True
        assert result.warnings == []

    def test_safe_command_allowlist_coverage(self, checker: SecurityChecker) -> None:
        for cmd in ["ls", "python", "docker", "npm", "cat", "pytest", "curl"]:
            parsed = ParsedCommand(command_names=[cmd], raw_command=cmd)
            result = checker.check_command(parsed)
            assert result.allowed is True, f"Safe command '{cmd}' was blocked"
            assert result.warnings == [], f"Safe command '{cmd}' produced warnings"


class TestUnknownCommands:
    def test_unknown_command_warns(self, checker: SecurityChecker) -> None:
        parsed = ParsedCommand(
            command_names=["my_custom_tool"],
            raw_command="my_custom_tool --flag",
        )
        result = checker.check_command(parsed)
        assert result.allowed is True
        assert len(result.warnings) >= 1
        assert "my_custom_tool" in result.warnings[0]

    def test_unknown_command_is_not_blocked(self, checker: SecurityChecker) -> None:
        parsed = ParsedCommand(
            command_names=["totally_unknown_bin"],
            raw_command="totally_unknown_bin --run",
        )
        result = checker.check_command(parsed)
        assert result.allowed is True


class TestRedirects:
    def test_dangerous_redirect_blocked(self, checker: SecurityChecker) -> None:
        parsed = ParsedCommand(
            command_names=["echo"],
            raw_command="echo hacked > /etc/passwd",
            redirects=[RedirectInfo(target_path="/etc/passwd", mode="write")],
        )
        result = checker.check_command(parsed)
        assert result.allowed is False
        assert "/etc/passwd" in result.reason

    def test_safe_redirect_allowed(self, checker: SecurityChecker) -> None:
        parsed = ParsedCommand(
            command_names=["echo"],
            raw_command="echo hello > /tmp/safe.txt",
            redirects=[RedirectInfo(target_path="/tmp/safe.txt", mode="write")],
        )
        result = checker.check_command(parsed)
        assert result.allowed is True


class TestPathChecks:
    def test_external_path_warning(self, checker: SecurityChecker) -> None:
        parsed = ParsedCommand(
            command_names=["cat"],
            file_paths=["/etc/passwd"],
            raw_command="cat /etc/passwd",
        )
        result = checker.check_command(parsed, project_root="/Users/sys/project")
        assert result.allowed is True
        assert any("outside project" in w for w in result.warnings)

    def test_internal_path_no_warning(self, checker: SecurityChecker) -> None:
        parsed = ParsedCommand(
            command_names=["cat"],
            file_paths=["/Users/sys/project/src/main.py"],
            raw_command="cat /Users/sys/project/src/main.py",
        )
        result = checker.check_command(parsed, project_root="/Users/sys/project")
        assert result.allowed is True
        assert result.warnings == []


class TestEnvSanitization:
    def test_env_sanitization_blocks_dangerous_keys(self) -> None:
        user_env = {
            "NODE_OPTIONS": "--require=./hack.js",
            "MY_VAR": "safe",
            "LD_PRELOAD": "/tmp/evil.so",
        }
        base_env = {"PATH": "/usr/bin"}
        result = sanitize_env(user_env, base_env)
        assert "MY_VAR" in result
        assert result["MY_VAR"] == "safe"
        assert "NODE_OPTIONS" not in result
        assert "LD_PRELOAD" not in result

    def test_env_path_override_rejected(self) -> None:
        user_env = {"PATH": "/evil/path", "HOME": "/tmp"}
        base_env = {"PATH": "/usr/bin"}
        result = sanitize_env(user_env, base_env)
        assert result["PATH"] == "/usr/bin"
        assert result["HOME"] == "/tmp"

    def test_env_blocked_by_prefix(self) -> None:
        user_env = {"DYLD_CUSTOM": "evil"}
        base_env = {"PATH": "/usr/bin"}
        result = sanitize_env(user_env, base_env)
        assert "DYLD_CUSTOM" not in result

    def test_env_all_blocked_keys_filtered(self) -> None:
        user_env = {key: "evil" for key in BLOCKED_ENV_KEYS}
        base_env = {"PATH": "/usr/bin"}
        result = sanitize_env(user_env, base_env)
        for key in BLOCKED_ENV_KEYS:
            assert key not in result

    def test_env_safe_keys_preserved(self) -> None:
        user_env = {"MY_APP_CONFIG": "/app/config", "DEBUG": "true", "PORT": "8080"}
        base_env = {"PATH": "/usr/bin", "HOME": "/home/user"}
        result = sanitize_env(user_env, base_env)
        assert result["MY_APP_CONFIG"] == "/app/config"
        assert result["DEBUG"] == "true"
        assert result["PORT"] == "8080"
        assert result["HOME"] == "/home/user"
