"""Tests for tree-sitter based bash command parser."""

from __future__ import annotations

import os

import pytest

from engine.tools.builtin._bash.ast_parser import BashASTParser, ParsedCommand


@pytest.fixture
def parser() -> BashASTParser:
    return BashASTParser()


class TestSimpleCommand:
    def test_extracts_command_name(self, parser: BashASTParser) -> None:
        result = parser.parse("ls -la /tmp")
        assert "ls" in result.command_names

    def test_extracts_file_path(self, parser: BashASTParser) -> None:
        result = parser.parse("ls -la /tmp")
        assert "/tmp" in result.file_paths

    def test_no_dynamic_parts(self, parser: BashASTParser) -> None:
        result = parser.parse("ls -la /tmp")
        assert result.has_dynamic_parts is False

    def test_flags_not_in_paths(self, parser: BashASTParser) -> None:
        result = parser.parse("ls -la /tmp")
        for p in result.file_paths:
            assert not p.startswith("-")


class TestPipedCommand:
    def test_detects_all_commands(self, parser: BashASTParser) -> None:
        result = parser.parse("cat file.txt | grep 'error' | wc -l")
        assert "cat" in result.command_names
        assert "grep" in result.command_names
        assert "wc" in result.command_names

    def test_has_pipes(self, parser: BashASTParser) -> None:
        result = parser.parse("cat file.txt | grep 'error' | wc -l")
        assert result.has_pipes is True

    def test_no_pipes_for_simple(self, parser: BashASTParser) -> None:
        result = parser.parse("echo hello")
        assert result.has_pipes is False


class TestRedirectWrite:
    def test_redirect_mode(self, parser: BashASTParser) -> None:
        result = parser.parse("echo hello > output.txt")
        assert len(result.redirects) >= 1
        assert result.redirects[0].mode == "write"

    def test_redirect_target(self, parser: BashASTParser) -> None:
        result = parser.parse("echo hello > output.txt")
        assert any("output.txt" in r.target_path for r in result.redirects)


class TestRedirectAppend:
    def test_append_mode(self, parser: BashASTParser) -> None:
        result = parser.parse("echo world >> log.txt")
        assert len(result.redirects) >= 1
        assert any(r.mode == "append" for r in result.redirects)

    def test_append_target(self, parser: BashASTParser) -> None:
        result = parser.parse("echo world >> log.txt")
        assert any("log.txt" in r.target_path for r in result.redirects)


class TestDynamicVariable:
    def test_home_variable_dynamic(self, parser: BashASTParser) -> None:
        result = parser.parse("echo $HOME/test")
        assert result.has_dynamic_parts is True


class TestCommandSubstitution:
    def test_substitution_detected(self, parser: BashASTParser) -> None:
        result = parser.parse("cat $(find /tmp -name '*.log')")
        assert result.has_dynamic_parts is True

    def test_substitution_inner_command(self, parser: BashASTParser) -> None:
        result = parser.parse("cat $(find /tmp -name '*.log')")
        assert "find" in result.command_names
        assert "cat" in result.command_names


class TestEmptyCommand:
    def test_empty_string(self, parser: BashASTParser) -> None:
        result = parser.parse("")
        assert result.command_names == []
        assert result.file_paths == []
        assert result.redirects == []
        assert result.has_pipes is False
        assert result.has_dynamic_parts is False

    def test_quotes_only(self, parser: BashASTParser) -> None:
        result = parser.parse('""')
        assert isinstance(result, ParsedCommand)


class TestWhitespaceCommand:
    def test_whitespace_only(self, parser: BashASTParser) -> None:
        result = parser.parse("   ")
        assert result.command_names == []
        assert result.file_paths == []
        assert result.redirects == []


class TestMultipleCommandsAnd:
    def test_list_operator(self, parser: BashASTParser) -> None:
        result = parser.parse("echo a && echo b")
        assert "echo" in result.command_names

    def test_list_both_sides(self, parser: BashASTParser) -> None:
        result = parser.parse("echo a && echo b")
        assert result.command_names.count("echo") == 2


class TestHomeExpansion:
    def test_tilde_expanded(self, parser: BashASTParser) -> None:
        result = parser.parse("cat ~/file.txt")
        home = os.path.expanduser("~")
        assert any(p.startswith(home) for p in result.file_paths)

    def test_tilde_only(self, parser: BashASTParser) -> None:
        result = parser.parse("cd ~")
        assert os.path.expanduser("~") in result.file_paths


class TestRedirectRead:
    def test_read_redirect(self, parser: BashASTParser) -> None:
        result = parser.parse("sort < input.txt")
        assert len(result.redirects) >= 1
        read_redirects = [r for r in result.redirects if r.mode == "read"]
        assert len(read_redirects) >= 1


class TestRawCommandPreserved:
    def test_raw_command_stored(self, parser: BashASTParser) -> None:
        cmd = "echo 'hello world'"
        result = parser.parse(cmd)
        assert result.raw_command == cmd


class TestUnquote:
    def test_double_quotes(self, parser: BashASTParser) -> None:
        assert BashASTParser._unquote('"hello"') == "hello"

    def test_single_quotes(self, parser: BashASTParser) -> None:
        assert BashASTParser._unquote("'hello'") == "hello"

    def test_no_quotes(self, parser: BashASTParser) -> None:
        assert BashASTParser._unquote("hello") == "hello"

    def test_mismatched_quotes(self, parser: BashASTParser) -> None:
        assert BashASTParser._unquote("'hello\"") == "'hello\""
