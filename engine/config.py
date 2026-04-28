import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from engine.providers.provider_models import ProviderConfig, resolve_model_ref


# Keys that must not appear in model params (reserved for LLM call construction)
RESERVED_MODEL_PARAM_KEYS = {"model", "messages", "tools"}

# Keys required for each provider entry in the "providers" dict
REQUIRED_PROVIDER_KEYS = {"api_key", "base_url"}


@dataclass
class Config:
    # Provider configuration — dict of provider name → ProviderConfig
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)

    # Primary model reference in "provider/model" format (required)
    primary: str = ""

    # Fallback model references in "provider/model" format (optional, defaults to [])
    fallback: List[str] = field(default_factory=list)

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

    # Timezone
    user_timezone: Optional[str] = None


class ConfigLoader:

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

        # Validate providers structure and build ProviderConfig instances
        providers = ConfigLoader._validate_and_build_providers(data, path)

        # Validate primary and fallback references
        primary = ConfigLoader._validate_primary(data, providers, path)
        fallback = ConfigLoader._validate_fallback(data, providers, path)

        # Build kwargs for Config from known fields only (unknown keys ignored)
        known_fields = {
            "providers",
            "primary",
            "fallback",
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
            "user_timezone",
        }

        kwargs = {k: v for k, v in data.items() if k in known_fields}
        # Override parsed providers/primary/fallback into kwargs
        kwargs["providers"] = providers
        kwargs["primary"] = primary
        kwargs["fallback"] = fallback
        config = Config(**kwargs)

        # Env var override for timezone (takes precedence over JSON)
        user_tz_env = os.environ.get("USER_TIMEZONE")
        if user_tz_env:
            config.user_timezone = user_tz_env

        return config

    @staticmethod
    def _validate_and_build_providers(
        data: dict, path: str
    ) -> Dict[str, ProviderConfig]:
        """Validate the 'providers' dict and build ProviderConfig instances.

        Checks:
        - 'providers' key exists and is a dict
        - Each provider has required keys (api_key, base_url)
        - Model params do not contain reserved keys

        Returns:
            Dict mapping provider name → ProviderConfig
        """
        if "providers" not in data:
            raise ValueError(f"Missing required key 'providers' in {path}")

        providers_raw = data["providers"]
        if not isinstance(providers_raw, dict):
            raise ValueError(f"'providers' must be a dict in {path}")

        if len(providers_raw) == 0:
            raise ValueError(f"'providers' must not be empty in {path}")

        providers: Dict[str, ProviderConfig] = {}
        for prov_name, prov_data in providers_raw.items():
            if not isinstance(prov_data, dict):
                raise ValueError(
                    f"Provider '{prov_name}' must be a dict in {path}"
                )

            missing = REQUIRED_PROVIDER_KEYS - set(prov_data.keys())
            if missing:
                raise ValueError(
                    f"Provider '{prov_name}' missing required keys "
                    f"{sorted(missing)} in {path}"
                )

            # Validate model params for reserved keys
            models_raw = prov_data.get("models", {})
            if not isinstance(models_raw, dict):
                raise ValueError(
                    f"Provider '{prov_name}' has invalid 'models' field "
                    f"(must be a dict) in {path}"
                )

            for model_name, model_params in models_raw.items():
                if not isinstance(model_params, dict):
                    raise ValueError(
                        f"Provider '{prov_name}' model '{model_name}' params "
                        f"must be a dict in {path}"
                    )
                reserved = RESERVED_MODEL_PARAM_KEYS & set(model_params.keys())
                if reserved:
                    raise ValueError(
                        f"Provider '{prov_name}' model '{model_name}' contains "
                        f"reserved keys {sorted(reserved)} in {path}. "
                        f"Keys {sorted(RESERVED_MODEL_PARAM_KEYS)} are reserved "
                        f"for LLM call construction"
                    )

            # Build ProviderConfig with defaults for optional fields
            providers[prov_name] = ProviderConfig(
                name=prov_name,
                api_key=prov_data["api_key"],
                base_url=prov_data["base_url"],
                rpm_limit=float(prov_data.get("rpm_limit", 100)),
                tpm_limit=float(prov_data.get("tpm_limit", 100000)),
                models=models_raw,
            )

        return providers

    @staticmethod
    def _validate_primary(
        data: dict, providers: Dict[str, ProviderConfig], path: str
    ) -> str:
        """Validate the 'primary' field.

        Checks:
        - 'primary' key exists
        - Format is 'provider/model'
        - Referenced provider and model exist

        Returns:
            The validated primary reference string
        """
        if "primary" not in data:
            raise ValueError(f"Missing required key 'primary' in {path}")

        primary = data["primary"]
        if not isinstance(primary, str) or not primary:
            raise ValueError(f"'primary' must be a non-empty string in {path}")

        try:
            prov_name, model_name = resolve_model_ref(primary)
        except ValueError as e:
            raise ValueError(f"Invalid 'primary' reference in {path}: {e}") from e

        if prov_name not in providers:
            raise ValueError(
                f"'primary' references unknown provider '{prov_name}' in {path}. "
                f"Available providers: {sorted(providers.keys())}"
            )

        if model_name not in providers[prov_name].models:
            raise ValueError(
                f"'primary' references unknown model '{model_name}' under "
                f"provider '{prov_name}' in {path}. "
                f"Available models: {sorted(providers[prov_name].models.keys())}"
            )

        return primary

    @staticmethod
    def _validate_fallback(
        data: dict, providers: Dict[str, ProviderConfig], path: str
    ) -> List[str]:
        """Validate the 'fallback' field.

        Checks for each entry:
        - Format is 'provider/model'
        - Referenced provider and model exist

        Returns:
            The validated fallback list (defaults to [] if absent)
        """
        fallback_raw = data.get("fallback", [])
        if fallback_raw is None:
            fallback_raw = []

        if not isinstance(fallback_raw, list):
            raise ValueError(f"'fallback' must be a list in {path}")

        fallback: List[str] = []
        for i, ref in enumerate(fallback_raw):
            if not isinstance(ref, str) or not ref:
                raise ValueError(
                    f"'fallback' entry at index {i} must be a non-empty "
                    f"string in {path}"
                )

            try:
                prov_name, model_name = resolve_model_ref(ref)
            except ValueError as e:
                raise ValueError(
                    f"Invalid 'fallback' entry at index {i} in {path}: {e}"
                ) from e

            if prov_name not in providers:
                raise ValueError(
                    f"'fallback' entry at index {i} references unknown "
                    f"provider '{prov_name}' in {path}. "
                    f"Available providers: {sorted(providers.keys())}"
                )

            if model_name not in providers[prov_name].models:
                raise ValueError(
                    f"'fallback' entry at index {i} references unknown model "
                    f"'{model_name}' under provider '{prov_name}' in {path}. "
                    f"Available models: {sorted(providers[prov_name].models.keys())}"
                )

            fallback.append(ref)

        return fallback


_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = ConfigLoader.load_from_json()
    return _config
