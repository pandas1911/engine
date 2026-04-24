import json
import os
import tempfile

import pytest

from engine.config import Config, ConfigLoader


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _valid_profile(overrides=None):
    profile = {
        "name": "test-profile",
        "api_key": "test-key",
        "base_url": "https://api.test.com/v1",
        "model": "test-model",
        "rpm_limit": 100,
        "tpm_limit": 0,
    }
    if overrides:
        profile.update(overrides)
    return profile


def _valid_config_data(overrides=None):
    data = {"provider_profiles": [_valid_profile()]}
    if overrides:
        data.update(overrides)
    return data


# ──────────────────────────────────────────────────────────────
# 1. Load from explicit JSON path
# ──────────────────────────────────────────────────────────────

def test_load_from_explicit_json_path_returns_valid_config():
    data = _valid_config_data()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(data, f)
        path = f.name

    try:
        config = ConfigLoader.load_from_json(path)
        assert isinstance(config, Config)
        assert len(config.provider_profiles) == 1
        profile = config.provider_profiles[0]
        assert profile["name"] == "test-profile"
        assert profile["api_key"] == "test-key"
        assert profile["base_url"] == "https://api.test.com/v1"
        assert profile["model"] == "test-model"
    finally:
        os.unlink(path)


# ──────────────────────────────────────────────────────────────
# 2. Auto-discovery finds engine.json
# ──────────────────────────────────────────────────────────────

def test_find_config_file_discovers_engine_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        engine_json = os.path.join(tmpdir, "engine.json")
        _write_json(engine_json, _valid_config_data())
        found = ConfigLoader.find_config_file(start_dir=tmpdir)
        assert found == engine_json


# ──────────────────────────────────────────────────────────────
# 3. Missing engine.json raises FileNotFoundError
# ──────────────────────────────────────────────────────────────

def test_missing_engine_json_raises_file_not_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError, match=tmpdir):
            ConfigLoader.find_config_file(start_dir=tmpdir)


def test_load_from_json_with_none_and_no_config_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            ConfigLoader.load_from_json(path=None)


# ──────────────────────────────────────────────────────────────
# 4. Malformed JSON raises ValueError with path
# ──────────────────────────────────────────────────────────────

def test_malformed_json_raises_value_error_with_path():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        f.write("{not valid json")
        path = f.name

    try:
        with pytest.raises(ValueError, match=path):
            ConfigLoader.load_from_json(path)
    finally:
        os.unlink(path)


# ──────────────────────────────────────────────────────────────
# 5. Missing provider_profiles key raises ValueError
# ──────────────────────────────────────────────────────────────

def test_missing_provider_profiles_raises_value_error():
    data = {"strip_thinking": False}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(data, f)
        path = f.name

    try:
        with pytest.raises(ValueError, match="provider_profiles"):
            ConfigLoader.load_from_json(path)
    finally:
        os.unlink(path)


# ──────────────────────────────────────────────────────────────
# 6. Empty provider_profiles raises ValueError
# ──────────────────────────────────────────────────────────────

def test_empty_provider_profiles_raises_value_error():
    data = {"provider_profiles": []}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(data, f)
        path = f.name

    try:
        with pytest.raises(ValueError, match="provider_profiles"):
            ConfigLoader.load_from_json(path)
    finally:
        os.unlink(path)


# ──────────────────────────────────────────────────────────────
# 7. Profile missing required key raises ValueError with key name
# ──────────────────────────────────────────────────────────────

def test_profile_missing_api_key_raises_value_error():
    profile = _valid_profile()
    del profile["api_key"]
    data = {"provider_profiles": [profile]}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(data, f)
        path = f.name

    try:
        with pytest.raises(ValueError, match="api_key"):
            ConfigLoader.load_from_json(path)
    finally:
        os.unlink(path)


def test_profile_missing_base_url_raises_value_error():
    profile = _valid_profile()
    del profile["base_url"]
    data = {"provider_profiles": [profile]}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(data, f)
        path = f.name

    try:
        with pytest.raises(ValueError, match="base_url"):
            ConfigLoader.load_from_json(path)
    finally:
        os.unlink(path)


def test_profile_missing_model_raises_value_error():
    profile = _valid_profile()
    del profile["model"]
    data = {"provider_profiles": [profile]}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(data, f)
        path = f.name

    try:
        with pytest.raises(ValueError, match="model"):
            ConfigLoader.load_from_json(path)
    finally:
        os.unlink(path)


# ──────────────────────────────────────────────────────────────
# 8. max_concurrent_agents < 2 raises ValueError
# ──────────────────────────────────────────────────────────────

def test_max_concurrent_agents_less_than_two_raises_value_error():
    data = _valid_config_data({"max_concurrent_agents": 1})
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(data, f)
        path = f.name

    try:
        with pytest.raises(ValueError, match="max_concurrent_agents"):
            ConfigLoader.load_from_json(path)
    finally:
        os.unlink(path)


# ──────────────────────────────────────────────────────────────
# 9. Config default values preserved
# ──────────────────────────────────────────────────────────────

def test_config_default_values():
    config = Config()
    assert config.provider_profiles == []
    assert config.strip_thinking is True
    assert config.max_depth == 4
    assert config.spawn_timeout == 60.0
    assert config.enable_wake_on_descendants is True
    assert config.max_concurrent_agents == 8
    assert config.agent_timeout == 300.0
    assert config.max_registry_size == 1000
    assert config.max_result_length == 2500
    assert config.summary_warning_reserve == 2
    assert config.emergency_summary_enabled is True
    assert config.emergency_summary_context_messages == 0
    assert config.log_dir is None
    assert config.rate_limit_rpm == 300.0
    assert config.rate_limit_burst == 3
    assert config.llm_retry_max_attempts == 3
    assert config.llm_retry_base_delay == 1.0
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

    # api_key, base_url, model should NOT be fields on Config
    assert not hasattr(config, "api_key")
    assert not hasattr(config, "base_url")
    assert not hasattr(config, "model")


# ──────────────────────────────────────────────────────────────
# 10. Optional fields from JSON override defaults
# ──────────────────────────────────────────────────────────────

def test_optional_fields_overridden_from_json():
    data = _valid_config_data({
        "strip_thinking": False,
        "max_depth": 10,
        "spawn_timeout": 120.0,
        "max_concurrent_agents": 16,
        "main_lane_concurrency": 8,
        "subagent_lane_concurrency": 16,
        "pacing_enabled": False,
        "pacing_min_interval_ms": 1000.0,
    })
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(data, f)
        path = f.name

    try:
        config = ConfigLoader.load_from_json(path)
        assert config.strip_thinking is False
        assert config.max_depth == 10
        assert config.spawn_timeout == 120.0
        assert config.max_concurrent_agents == 16
        assert config.main_lane_concurrency == 8
        assert config.subagent_lane_concurrency == 16
        assert config.pacing_enabled is False
        assert config.pacing_min_interval_ms == 1000.0
    finally:
        os.unlink(path)


# ──────────────────────────────────────────────────────────────
# 11. Unknown top-level keys silently ignored
# ──────────────────────────────────────────────────────────────

def test_unknown_top_level_keys_ignored():
    data = _valid_config_data({"future_field": 123, "another_unknown": "hello"})
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(data, f)
        path = f.name

    try:
        config = ConfigLoader.load_from_json(path)
        assert isinstance(config, Config)
        assert len(config.provider_profiles) == 1
    finally:
        os.unlink(path)


# ──────────────────────────────────────────────────────────────
# 12. Non-UTF-8 file raises clear error
# ──────────────────────────────────────────────────────────────

def test_non_utf8_file_raises_clear_error():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        f.write(b'\xff\xfe{"provider_profiles": []}')
        path = f.name

    try:
        with pytest.raises((ValueError, UnicodeDecodeError)):
            ConfigLoader.load_from_json(path)
    finally:
        os.unlink(path)


# ──────────────────────────────────────────────────────────────
# 13. Unreadable file raises clear error
# ──────────────────────────────────────────────────────────────

def test_unreadable_file_raises_clear_error():
    data = _valid_config_data()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(data, f)
        path = f.name

    try:
        os.chmod(path, 0o000)
        with pytest.raises((PermissionError, OSError)):
            ConfigLoader.load_from_json(path)
    finally:
        os.chmod(path, 0o644)
        os.unlink(path)


# ──────────────────────────────────────────────────────────────
# ConfigLoader._validate_profiles direct tests
# ──────────────────────────────────────────────────────────────

def test_validate_profiles_accepts_valid_profiles():
    data = {"provider_profiles": [_valid_profile(), _valid_profile({"name": "second"})]}
    ConfigLoader._validate_profiles(data, "/tmp/fake.json")


def test_validate_profiles_rejects_non_list():
    data = {"provider_profiles": "not-a-list"}
    with pytest.raises(ValueError, match="provider_profiles"):
        ConfigLoader._validate_profiles(data, "/tmp/fake.json")


def test_validate_profiles_rejects_empty_list():
    data = {"provider_profiles": []}
    with pytest.raises(ValueError, match="provider_profiles"):
        ConfigLoader._validate_profiles(data, "/tmp/fake.json")
