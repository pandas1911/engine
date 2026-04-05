from dataclasses import dataclass
from dotenv import load_dotenv
import os


@dataclass
class Config:
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    max_depth: int = 3
    default_model: str = "MiniMax-M2.7"
    spawn_timeout: float = 60.0
    enable_wake_on_descendants: bool = True
    max_concurrent_agents: int = 10
    agent_timeout: float = 300.0
    max_registry_size: int = 1000
    max_result_length: int = 4000


class ConfigLoader:
    REQUIRED_KEYS = ["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"]

    @staticmethod
    def load_from_env() -> Config:
        load_dotenv()

        missing_keys = []
        config_values = {}

        for key in ConfigLoader.REQUIRED_KEYS:
            value = os.getenv(key)
            if value is None or value.strip() == "":
                missing_keys.append(key)
            else:
                config_values[key.lower()] = value

        if missing_keys:
            raise ValueError(
                f"Missing required configuration keys: {', '.join(missing_keys)}. "
                f"Please ensure these are set in your .env file or environment."
            )

        return Config(
            openai_api_key=config_values["openai_api_key"],
            openai_base_url=config_values["openai_base_url"],
            openai_model=config_values["openai_model"],
        )


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = ConfigLoader.load_from_env()
    return _config
