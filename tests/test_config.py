import pytest
from engine.config import Config, ConfigProvider, ConfigLoader


class MockConfigProvider(ConfigProvider):
    def __init__(self, values=None):
        self.values = values or {
            "LLM_API_KEY": "test-key",
            "LLM_BASE_URL": "http://test.example.com",
            "LLM_MODEL": "test-model",
        }

    def get(self, key):
        return self.values.get(key)


def test_legacy_env_creates_default_provider_profile():
    provider = MockConfigProvider()
    config = ConfigLoader.load(provider)
    assert len(config.provider_profiles) == 1
    profile = config.provider_profiles[0]
    assert profile["name"] == "default"
    assert profile["api_key"] == "test-key"
    assert profile["base_url"] == "http://test.example.com"
    assert profile["model"] == "test-model"
    assert profile["rpm_limit"] == 300.0
    assert profile["tpm_limit"] == 0


def test_json_provider_profiles_parsed_correctly():
    profiles_json = '[{"name": "openai", "api_key": "key1", "base_url": "https://api.openai.com", "model": "gpt-4", "rpm_limit": 100, "tpm_limit": 50000}]'
    provider = MockConfigProvider(
        values={
            "LLM_API_KEY": "test-key",
            "LLM_BASE_URL": "http://test.example.com",
            "LLM_MODEL": "test-model",
            "LLM_PROVIDER_PROFILES": profiles_json,
        }
    )
    config = ConfigLoader.load(provider)
    assert len(config.provider_profiles) == 1
    profile = config.provider_profiles[0]
    assert profile["name"] == "openai"
    assert profile["api_key"] == "key1"
    assert profile["base_url"] == "https://api.openai.com"
    assert profile["model"] == "gpt-4"
    assert profile["rpm_limit"] == 100
    assert profile["tpm_limit"] == 50000


def test_new_config_fields_have_correct_defaults():
    provider = MockConfigProvider()
    config = ConfigLoader.load(provider)
    assert config.main_lane_concurrency == 4
    assert config.subagent_lane_concurrency == 8
    assert config.pacing_enabled is True
    assert config.pacing_min_interval_ms == 500.0
    assert config.api_queue_max_size == 100
    assert config.api_queue_timeout == 120.0
    assert config.key_rotation_enabled is True
    assert config.fallback_enabled is True
    assert config.cooldown_initial_ms == 30000.0
    assert config.cooldown_max_ms == 300000.0


def test_lane_concurrency_overridden_from_env():
    provider = MockConfigProvider(
        values={
            "LLM_API_KEY": "test-key",
            "LLM_BASE_URL": "http://test.example.com",
            "LLM_MODEL": "test-model",
            "LLM_MAIN_LANE_CONCURRENCY": "10",
            "LLM_SUBAGENT_LANE_CONCURRENCY": "16",
        }
    )
    config = ConfigLoader.load(provider)
    assert config.main_lane_concurrency == 10
    assert config.subagent_lane_concurrency == 16


def test_pacing_settings_overridden_from_env():
    provider = MockConfigProvider(
        values={
            "LLM_API_KEY": "test-key",
            "LLM_BASE_URL": "http://test.example.com",
            "LLM_MODEL": "test-model",
            "LLM_PACING_ENABLED": "false",
            "LLM_PACING_MIN_INTERVAL_MS": "1000",
        }
    )
    config = ConfigLoader.load(provider)
    assert config.pacing_enabled is False
    assert config.pacing_min_interval_ms == 1000.0


def test_bool_env_variations():
    for val in ("true", "1", "yes", "on", "TRUE", "True"):
        provider = MockConfigProvider(
            values={
                "LLM_API_KEY": "test-key",
                "LLM_BASE_URL": "http://test.example.com",
                "LLM_MODEL": "test-model",
                "LLM_PACING_ENABLED": val,
            }
        )
        config = ConfigLoader.load(provider)
        assert config.pacing_enabled is True, f"failed for {val!r}"

    for val in ("false", "0", "no", "off", "FALSE", "False"):
        provider = MockConfigProvider(
            values={
                "LLM_API_KEY": "test-key",
                "LLM_BASE_URL": "http://test.example.com",
                "LLM_MODEL": "test-model",
                "LLM_PACING_ENABLED": val,
            }
        )
        config = ConfigLoader.load(provider)
        assert config.pacing_enabled is False, f"failed for {val!r}"


def test_invalid_bool_raises_value_error():
    provider = MockConfigProvider(
        values={
            "LLM_API_KEY": "test-key",
            "LLM_BASE_URL": "http://test.example.com",
            "LLM_MODEL": "test-model",
            "LLM_PACING_ENABLED": "maybe",
        }
    )
    with pytest.raises(ValueError, match="expected bool"):
        ConfigLoader.load(provider)


def test_invalid_json_raises_value_error():
    provider = MockConfigProvider(
        values={
            "LLM_API_KEY": "test-key",
            "LLM_BASE_URL": "http://test.example.com",
            "LLM_MODEL": "test-model",
            "LLM_PROVIDER_PROFILES": "not-json",
        }
    )
    with pytest.raises(ValueError, match="expected JSON"):
        ConfigLoader.load(provider)


def test_required_keys_still_validated():
    provider = MockConfigProvider(
        values={
            "LLM_BASE_URL": "http://test.example.com",
            "LLM_MODEL": "test-model",
        }
    )
    with pytest.raises(ValueError, match="Missing required configuration keys: LLM_API_KEY"):
        ConfigLoader.load(provider)


def test_existing_optional_keys_still_work():
    provider = MockConfigProvider(
        values={
            "LLM_API_KEY": "test-key",
            "LLM_BASE_URL": "http://test.example.com",
            "LLM_MODEL": "test-model",
            "LLM_RATE_LIMIT_RPM": "600",
            "LLM_RETRY_MAX_ATTEMPTS": "5",
            "LLM_RETRY_BASE_DELAY": "2.0",
        }
    )
    config = ConfigLoader.load(provider)
    assert config.rate_limit_rpm == 600.0
    assert config.llm_retry_max_attempts == 5
    assert config.llm_retry_base_delay == 2.0
    assert config.provider_profiles[0]["rpm_limit"] == 600.0
