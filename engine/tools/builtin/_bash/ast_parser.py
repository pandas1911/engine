"""tree-sitter based bash command parser for security analysis."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import tree_sitter_bash as tsbash
from tree_sitter import Language, Parser, Node


@dataclass
class RedirectInfo:
    """A file redirect found in a command."""
    target_path: str
    mode: str  # "write" (>) | "append" (>>) | "read" (<)


@dataclass
class ParsedCommand:
    """Structured result of parsing a bash command."""
    command_names: List[str] = field(default_factory=list)
    file_paths: List[str] = field(default_factory=list)
    redirects: List[RedirectInfo] = field(default_factory=list)
    has_pipes: bool = False
    has_dynamic_parts: bool = False
    raw_command: str = ""


class BashASTParser:
    """Parse bash commands using tree-sitter and extract security-relevant info."""

    def __init__(self) -> None:
        self._parser: Optional[Parser] = None

    def _ensure_parser(self) -> Parser:
        """Lazy-initialize the tree-sitter parser with bash grammar."""
        if self._parser is None:
            language = Language(tsbash.language())
            self._parser = Parser(language)
        return self._parser

    def parse(self, command: str) -> ParsedCommand:
        """Parse a bash command string into a structured ParsedCommand."""
        if not command or not command.strip():
            return ParsedCommand(raw_command=command)

        parser = self._ensure_parser()
        tree = parser.parse(command.encode("utf-8"))
        if not tree or not tree.root_node:
            return ParsedCommand(raw_command=command)

        root = tree.root_node
        result = ParsedCommand(raw_command=command)
        result.has_pipes = self._has_pipeline(root)
        self._extract_from_node(root, result)
        return result

    def _has_pipeline(self, root: Node) -> bool:
        """Check if the AST contains a pipeline node."""
        for i in range(root.child_count):
            child = root.child(i)
            if child and child.type == "pipeline":
                return True
        return False

    def _extract_from_node(self, node: Node, result: ParsedCommand) -> None:
        """Recursively extract command names, paths, redirects from AST."""
        self._walk(node, result)

    def _walk(self, node: Node, result: ParsedCommand) -> None:
        """Walk the entire tree, extracting relevant nodes."""
        if node.type == "command":
            self._extract_command(node, result)
        elif node.type == "file_redirect":
            redirect = self._parse_redirect(node)
            if redirect:
                result.redirects.append(redirect)
        elif node.type in ("command_substitution", "process_substitution"):
            result.has_dynamic_parts = True
        elif node.type == "simple_expansion":
            # Only mark dynamic if it's not a standalone $HOME
            text = node.text.decode("utf-8")
            if text != "$HOME":
                result.has_dynamic_parts = True
        elif node.type == "expansion":
            result.has_dynamic_parts = True

        for child in node.children:
            self._walk(child, result)

    def _extract_command(self, cmd_node: Node, result: ParsedCommand) -> None:
        """Extract info from a single command node."""
        for child in cmd_node.children:
            if child.type == "command_name":
                result.command_names.append(child.text.decode("utf-8"))
            elif child.type == "command_substitution":
                result.has_dynamic_parts = True
            elif child.type == "concatenation":
                self._extract_concatenation(child, result)
            elif child.type in ("word", "string", "raw_string"):
                text = child.text.decode("utf-8")
                if self._is_dynamic(text):
                    result.has_dynamic_parts = True
                else:
                    text_unquoted = self._unquote(text)
                    if text_unquoted and not text_unquoted.startswith("-"):
                        expanded = self._expand_home(text_unquoted)
                        if expanded:
                            result.file_paths.append(expanded)

    def _extract_concatenation(self, node: Node, result: ParsedCommand) -> None:
        """Extract from concatenation node, detecting dynamic parts."""
        has_expansion = any(
            c.type in ("simple_expansion", "expansion", "command_substitution")
            for c in node.children
        )
        if has_expansion:
            result.has_dynamic_parts = True
        else:
            text = self._unquote(node.text.decode("utf-8"))
            if text and not text.startswith("-"):
                expanded = self._expand_home(text)
                if expanded:
                    result.file_paths.append(expanded)

    def _parse_redirect(self, node: Node) -> Optional[RedirectInfo]:
        """Parse a file_redirect node into RedirectInfo."""
        node_text = node.text.decode("utf-8")
        if ">>" in node_text:
            mode = "append"
        elif ">" in node_text:
            mode = "write"
        else:
            mode = "read"

        for child in node.children:
            if child.type in ("word", "string", "raw_string", "file_descriptor"):
                target = self._unquote(child.text.decode("utf-8"))
                return RedirectInfo(
                    target_path=self._expand_home(target),
                    mode=mode,
                )
        return None

    @staticmethod
    def _is_dynamic(text: str) -> bool:
        """Detect dynamic shell constructs that prevent static analysis."""
        return bool(
            "$(" in text
            or "${" in text
            or "`" in text
            or (text.startswith("$") and not text.startswith("$HOME"))
        )

    @staticmethod
    def _unquote(text: str) -> str:
        """Remove surrounding quotes from a string."""
        if len(text) >= 2:
            first, last = text[0], text[-1]
            if first in ('"', "'") and first == last:
                return text[1:-1]
        return text

    @staticmethod
    def _expand_home(text: str) -> str:
        """Expand ~ to home directory."""
        if text == "~":
            return os.path.expanduser("~")
        if text.startswith("~/") or text.startswith("~\\"):
            return os.path.expanduser("~") + text[1:]
        return text
