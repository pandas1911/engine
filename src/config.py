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
    max_depth: int = 3
    spawn_timeout: float = 60.0
    enable_wake_on_descendants: bool = True
    max_concurrent_agents: int = 10
    agent_timeout: float = 300.0
    max_registry_size: int = 1000
    max_result_length: int = 4000


class ConfigLoader:
    REQUIRED_KEYS = ["LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"]

    @staticmethod
    def load_from_env() -> Config:
        load_dotenv()

        missing_keys = []
        config_values = {}

        mapping = {
            "LLM_API_KEY": "api_key",
            "LLM_BASE_URL": "base_url",
            "LLM_MODEL": "model",
        }

        for key in ConfigLoader.REQUIRED_KEYS:
            value = os.getenv(key)
            if value is None or value.strip() == "":
                missing_keys.append(key)
            else:
                config_values[mapping[key]] = value

        if missing_keys:
            raise ValueError(
                f"Missing required configuration keys: {', '.join(missing_keys)}. "
                f"Please ensure these are set in your .env file or environment."
            )

        return Config(
            api_key=config_values["api_key"],
            base_url=config_values["base_url"],
            model=config_values["model"],
        )


_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = ConfigLoader.load_from_env()
    return _config
