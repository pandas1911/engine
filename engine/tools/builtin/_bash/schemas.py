"""Parameter schemas and constants for the bash tool package."""

# Execution Defaults
DEFAULT_TIMEOUT_MS: int = 120_000       # 2 minutes
MAX_OUTPUT_LINES: int = 2_000
MAX_OUTPUT_BYTES: int = 50 * 1024       # 50 KB
MAX_OUTPUT_SLIDING_WINDOW: int = MAX_OUTPUT_BYTES * 2  # 100 KB buffer
YIELD_THRESHOLD_MS: int = 10_000        # Auto-background after 10s
KILL_GRACE_PERIOD_MS: int = 3_000       # SIGTERM -> SIGKILL grace

# BashTool Parameter Schema (OpenAI function calling format)
BASH_TOOL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "The bash command to execute.",
        },
        "description": {
            "type": "string",
            "description": (
                "Clear, concise description of what this command does "
                "in 5-10 words. Example: 'List files in temp directory'."
            ),
        },
        "timeout": {
            "type": "number",
            "description": (
                "Optional timeout in milliseconds. Default: 120000 (2 min). "
                "Increase for long-running commands like builds or tests."
            ),
        },
        "workdir": {
            "type": "string",
            "description": (
                "The working directory for the command. "
                "Defaults to the project root directory. "
                "Use this instead of 'cd' in commands."
            ),
        },
        "env": {
            "type": "object",
            "description": (
                "Optional additional environment variables as key-value pairs. "
                "Dangerous keys (NODE_OPTIONS, LD_PRELOAD, etc.) are automatically blocked."
            ),
            "additionalProperties": {"type": "string"},
        },
        "background": {
            "type": "boolean",
            "description": (
                "Run the command in the background. "
                "Returns immediately with a session_id for later management "
                "via the 'process' tool. Default: false."
            ),
        },
    },
    "required": ["command", "description"],
}

# ProcessTool Parameter Schema
PROCESS_TOOL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["list", "poll", "log", "kill"],
            "description": (
                "Action to perform: "
                "'list' -- show all background processes; "
                "'poll' -- check status of a specific process; "
                "'log' -- retrieve output of a process; "
                "'kill' -- terminate a running process."
            ),
        },
        "session_id": {
            "type": "string",
            "description": "Background process session ID (required for poll, log, kill).",
        },
    },
    "required": ["action"],
}
