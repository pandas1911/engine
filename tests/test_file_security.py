"""Tests for PathGuard — deny-list file access control."""

import pytest
from engine.tools.builtin.security import PathGuard, DEFAULT_DENIED_PATTERNS


class TestPathGuardDefaults:
    """Tests with default denied patterns."""

    def test_env_file_denied(self):
        g = PathGuard()
        assert not g.is_path_allowed("/home/user/project/.env")
        assert "denied pattern" in g.check_path("/home/user/project/.env")

    def test_env_local_denied(self):
        g = PathGuard()
        assert not g.is_path_allowed("/home/user/project/.env.local")

    def test_git_internal_denied(self):
        g = PathGuard()
        assert not g.is_path_allowed("/project/.git/config")
        assert not g.is_path_allowed("/project/.git/HEAD")

    def test_credentials_denied(self):
        g = PathGuard()
        assert not g.is_path_allowed("/home/user/credentials.json")

    def test_ssh_key_denied(self):
        g = PathGuard()
        assert not g.is_path_allowed("/home/user/.ssh/id_rsa")
        assert not g.is_path_allowed("/home/user/.ssh/id_ed25519")

    def test_shadow_denied(self):
        g = PathGuard()
        assert not g.is_path_allowed("/etc/shadow")

    def test_normal_file_allowed(self):
        g = PathGuard()
        assert g.is_path_allowed("/home/user/project/main.py")
        assert g.is_path_allowed("/etc/hosts")

    def test_check_path_returns_none_for_allowed(self):
        g = PathGuard()
        assert g.check_path("/home/user/code/readme.md") is None


class TestPathGuardCustom:
    """Tests with custom denied patterns."""

    def test_custom_patterns_replace_defaults(self):
        g = PathGuard(denied_patterns=["**/secret/**"])
        # Default patterns are gone — .env is now allowed
        assert g.is_path_allowed("/home/user/project/.env")
        # Custom pattern works
        assert not g.is_path_allowed("/path/secret/data.txt")

    def test_empty_patterns_allows_all(self):
        g = PathGuard(denied_patterns=[])
        assert g.is_path_allowed("/anything/.env")

    def test_path_normalization_dotdot(self):
        g = PathGuard(denied_patterns=["**/secret/**"])
        # Path traversal through .. should still be caught after normalization
        assert not g.is_path_allowed("/home/user/secret/../secret/file.txt")
