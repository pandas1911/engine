from abc import ABC, abstractmethod
from dataclasses import dataclass
from dotenv import load_dotenv
import os
from typing import Optional


@dataclass
class Config:
    api_key: str
    base_url: str
    model: str
    strip_thinking: bool = True
    max_depth: int = 4
    spawn_timeout: float = 60.0
    enable_wake_on_descendants: bool = True
    max_concurrent_agents: int = 8
    agent_timeout: float = 300.0
    max_registry_size: int = 1000
    max_result_length: int = 2500
    # Iteration guard configuration
    summary_warning_reserve: int = 2  # iterations remaining when warning is injected (0 = disabled)
    emergency_summary_enabled: bool = True  # force a final summary call when loop exhausts
    emergency_summary_context_messages: int = 0  # messages to keep for emergency summary (0 = use full session)
    log_dir: Optional[str] = None

    # Rate limiting
    rate_limit_rpm: float = 300.0      # Requests per minute (0 = disabled)
    rate_limit_burst: int = 3           # Max concurrent API calls (token bucket burst capacity)

    # Retry
    llm_retry_max_attempts: int = 3     # Max retry attempts (1 = no retry, just the initial call)
    llm_retry_base_delay: float = 1.0   # Base delay in seconds for exponential backoff


class ConfigProvider(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        pass


class DotEnvProvider(ConfigProvider):
    def __init__(self, path: Optional[str] = None):
        target = path or os.path.join(os.getcwd(), ".env")
        if os.path.isfile(target):
            load_dotenv(target, override=False)

    def get(self, key: str) -> Optional[str]:
        return os.getenv(key)


class EnvVarProvider(ConfigProvider):
    def get(self, key: str) -> Optional[str]:
        return os.getenv(key)


class ConfigLoader:
    REQUIRED_KEYS = ["LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"]

    MAPPING = {
        "LLM_API_KEY": "api_key",
        "LLM_BASE_URL": "base_url",
        "LLM_MODEL": "model",
        "LOG_DIR": "log_dir",
    }

    OPTIONAL_INT_KEYS = {
        "LLM_RETRY_MAX_ATTEMPTS": "llm_retry_max_attempts",
        "LLM_RATE_LIMIT_BURST": "rate_limit_burst",
    }

    OPTIONAL_FLOAT_KEYS = {
        "LLM_RATE_LIMIT_RPM": "rate_limit_rpm",
        "LLM_RETRY_BASE_DELAY": "llm_retry_base_delay",
    }

    @staticmethod
    def load(provider: ConfigProvider) -> Config:
        missing_keys = []
        config_values = {}

        for key in ConfigLoader.REQUIRED_KEYS:
            value = provider.get(key)
            if value is None or value.strip() == "":
                missing_keys.append(key)
            else:
                config_values[ConfigLoader.MAPPING[key]] = value

        if missing_keys:
            raise ValueError(
                f"Missing required configuration keys: {', '.join(missing_keys)}. "
                f"Please ensure these are set in your .env file or environment."
            )

        config = Config(
            api_key=config_values["api_key"],
            base_url=config_values["base_url"],
            model=config_values["model"],
        )

        log_dir = provider.get("LOG_DIR")
        if log_dir:
            config.log_dir = log_dir

        for env_key, field_name in ConfigLoader.OPTIONAL_INT_KEYS.items():
            raw = provider.get(env_key)
            if raw is not None and raw.strip() != "":
                try:
                    setattr(config, field_name, int(raw))
                except ValueError:
                    raise ValueError(
                        "Invalid value for {}: expected int, got '{}'".format(env_key, raw)
                    )

        for env_key, field_name in ConfigLoader.OPTIONAL_FLOAT_KEYS.items():
            raw = provider.get(env_key)
            if raw is not None and raw.strip() != "":
                try:
                    setattr(config, field_name, float(raw))
                except ValueError:
                    raise ValueError(
                        "Invalid value for {}: expected float, got '{}'".format(env_key, raw)
                    )

        if config.max_concurrent_agents < 2:
            raise ValueError(
                f"max_concurrent_agents must be >= 2, got {config.max_concurrent_agents}. "
                "Values less than 2 can cause deadlock in the agent execution system."
            )

        return config

    @staticmethod
    def load_from_env(dotenv_path: Optional[str] = None) -> Config:
        if dotenv_path or os.getenv("LLM_API_KEY") is None:
            provider = DotEnvProvider(path=dotenv_path)
        else:
            provider = EnvVarProvider()
        return ConfigLoader.load(provider)


_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = ConfigLoader.load_from_env()
    return _config
