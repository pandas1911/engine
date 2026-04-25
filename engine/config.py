import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Config:
    # Provider profiles — each entry: {name, api_key, base_url, model, rpm_limit, tpm_limit, weight?}
    # Must contain at least one profile (validated at load time)
    provider_profiles: List[Dict] = field(default_factory=list)

    # LLM response processing
    strip_thinking: bool = True

    # Agent hierarchy
    max_depth: int = 3
    spawn_timeout: float = 60.0
    max_result_length: int = 3000

    # Iteration guard
    summary_warning_reserve: int = 2
    emergency_summary_enabled: bool = True
    emergency_summary_context_messages: int = 0

    # Logging
    log_dir: Optional[str] = None

    # Retry
    llm_retry_max_attempts: int = 3
    llm_retry_base_delay: float = 1.0

    # Lane concurrency
    main_lane_concurrency: int = 4
    subagent_lane_concurrency: int = 5

    # Adaptive pacing
    pacing_enabled: bool = True
    pacing_min_interval_ms: float = 500.0

    # Key rotation and fallback
    key_rotation_enabled: bool = True
    fallback_enabled: bool = True

    # Cooldown
    cooldown_initial_ms: float = 30000.0
    cooldown_max_ms: float = 300000.0


class ConfigLoader:
    REQUIRED_PROFILE_KEYS = {"name", "api_key", "base_url", "model"}

    @staticmethod
    def find_config_file(start_dir: Optional[str] = None) -> str:
        """Auto-discover engine.json: CWD → upward to pyproject.toml directory.

        Search strategy:
        1. If start_dir is provided, start there
        2. Otherwise start from CWD
        3. Look for engine.json in current directory
        4. Walk upward looking for pyproject.toml, then check that directory for engine.json
        5. Return absolute path if found, raise FileNotFoundError otherwise
        """
        current = os.path.abspath(start_dir) if start_dir else os.getcwd()

        # Check current directory first
        engine_json = os.path.join(current, "engine.json")
        if os.path.isfile(engine_json):
            return engine_json

        # Walk upward looking for pyproject.toml
        search_root = current
        while True:
            parent = os.path.dirname(search_root)
            if parent == search_root:
                break
            search_root = parent

            if os.path.isfile(os.path.join(search_root, "pyproject.toml")):
                engine_json = os.path.join(search_root, "engine.json")
                if os.path.isfile(engine_json):
                    return engine_json

        raise FileNotFoundError(
            f"No engine.json found starting from {current}"
        )

    @staticmethod
    def load_from_json(path: Optional[str] = None) -> Config:
        """Load Config from JSON file.

        Args:
            path: Explicit path to JSON file. If None, auto-discover.

        Returns:
            Config instance with values from JSON

        Raises:
            FileNotFoundError: If no config file found
            ValueError: If validation fails (malformed JSON, missing fields, etc.)
            PermissionError: If file is not readable
        """
        if path is None:
            path = ConfigLoader.find_config_file()

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path}: {e}") from e
        except PermissionError as e:
            raise PermissionError(f"Cannot read config file {path}: {e}") from e

        ConfigLoader._validate_profiles(data, path)

        # Build kwargs for Config from known fields only (unknown keys ignored)
        known_fields = {
            "provider_profiles",
            "strip_thinking",
            "max_depth",
            "spawn_timeout",
            "max_result_length",
            "summary_warning_reserve",
            "emergency_summary_enabled",
            "emergency_summary_context_messages",
            "log_dir",
            "llm_retry_max_attempts",
            "llm_retry_base_delay",
            "main_lane_concurrency",
            "subagent_lane_concurrency",
            "pacing_enabled",
            "pacing_min_interval_ms",
            "key_rotation_enabled",
            "fallback_enabled",
            "cooldown_initial_ms",
            "cooldown_max_ms",
        }

        kwargs = {k: v for k, v in data.items() if k in known_fields}
        config = Config(**kwargs)

        return config

    @staticmethod
    def _validate_profiles(data: dict, path: str) -> None:
        """Validate provider_profiles structure.

        Checks:
        - provider_profiles key exists
        - provider_profiles is a list
        - provider_profiles is non-empty
        - Each profile has all required keys (name, api_key, base_url, model)

        Args:
            data: Parsed JSON data
            path: File path (for error messages)

        Raises:
            ValueError: If any validation fails
        """
        if "provider_profiles" not in data:
            raise ValueError(f"Missing required key 'provider_profiles' in {path}")

        profiles = data["provider_profiles"]
        if not isinstance(profiles, list):
            raise ValueError(f"'provider_profiles' must be a list in {path}")

        if len(profiles) == 0:
            raise ValueError(f"'provider_profiles' must not be empty in {path}")

        for i, profile in enumerate(profiles):
            missing = ConfigLoader.REQUIRED_PROFILE_KEYS - set(profile.keys())
            if missing:
                raise ValueError(
                    f"Profile at index {i} missing required keys {sorted(missing)} in {path}"
                )


_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = ConfigLoader.load_from_json()
    return _config
