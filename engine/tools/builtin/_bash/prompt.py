"""LLM-facing description for the bash tool."""

BASH_TOOL_DESCRIPTION: str = """\
Execute bash shell commands on the local machine.

USAGE NOTES:
- Use this tool for running shell commands, scripts, and build tools.
- Avoid using bash for file reading -- use the read tool instead.
- Avoid using bash for file editing -- use the edit tool instead.
- Use the workdir parameter instead of 'cd' in commands.
- Each command requires a clear description of what it does (5-10 words).

TIMEOUT:
- Default timeout is 120000ms (2 minutes).
- Increase timeout for long-running commands like builds, tests, or installs.

OUTPUT:
- Output is truncated at 2000 lines or 50KB.
- Truncated output is saved to a file; the path is included in the response.

BACKGROUND EXECUTION:
- Set background=true to run commands in the background.
- Background commands return immediately with a session_id.
- Use the 'process' tool to list, poll, log, or kill background processes.

PLATFORM:
- Commands run on macOS with zsh/bash.
- Use Unix-style paths (forward slashes).

GIT SAFETY:
- Never use 'rm -rf /' or similar destructive commands.
- Be cautious with git commands that rewrite history (force push, rebase).
"""
