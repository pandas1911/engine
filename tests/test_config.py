"""Comprehensive config validation tests for engine.config.ConfigLoader.

Tests cover all edge cases for the new provider configuration structure,
including valid configs, missing keys, invalid references, reserved
model params, and resolve_model_ref behavior.
"""

import json
import os
import tempfile

import pytest

from engine.config import Config, ConfigLoader
from engine.providers.provider_models import resolve_model_ref


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_temp_config(data: dict) -> str:
    """Write *data* as JSON to a temporary file and return the path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(data, f)
    f.close()
    return f.name


def _valid_two_provider_config() -> dict:
    """Return a valid config dict with two providers."""
    return {
        "providers": {
            "aliyun": {
                "api_key": "sk-test-aliyun",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "models": {
                    "deepseek-v4-pro": {},
                    "claude-3.5-sonnet": {},
                },
            },
            "openrouter": {
                "api_key": "sk-test-openrouter",
                "base_url": "https://openrouter.ai/api/v1",
                "models": {
                    "gpt-4o": {"temperature": 0.7},
                },
            },
        },
        "primary": "aliyun/deepseek-v4-pro",
        "fallback": ["aliyun/claude-3.5-sonnet", "openrouter/gpt-4o"],
    }


# ---------------------------------------------------------------------------
# 1. Valid config loads
# ---------------------------------------------------------------------------

def test_valid_config_loads():
    """Valid new-format config with 2 providers loads successfully."""
    cfg_data = _valid_two_provider_config()
    path = _write_temp_config(cfg_data)
    try:
        config = ConfigLoader.load_from_json(path)
        assert isinstance(config, Config)
        assert len(config.providers) == 2
        assert "aliyun" in config.providers
        assert "openrouter" in config.providers
        assert config.primary == "aliyun/deepseek-v4-pro"
        assert len(config.fallback) == 2
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 2-3. Missing / empty providers
# ---------------------------------------------------------------------------

def test_missing_providers_key():
    """Config without 'providers' key raises ValueError mentioning 'providers'."""
    path = _write_temp_config({"primary": "aliyun/model"})
    try:
        with pytest.raises(ValueError, match="providers"):
            ConfigLoader.load_from_json(path)
    finally:
        os.unlink(path)


def test_empty_providers_dict():
    """Config with empty 'providers' dict raises ValueError."""
    path = _write_temp_config({"providers": {}, "primary": "aliyun/model"})
    try:
        with pytest.raises(ValueError, match="must not be empty"):
            ConfigLoader.load_from_json(path)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 4-6. Primary validation
# ---------------------------------------------------------------------------

def test_missing_primary():
    """Config without 'primary' key raises ValueError mentioning 'primary'."""
    cfg = _valid_two_provider_config()
    del cfg["primary"]
    path = _write_temp_config(cfg)
    try:
        with pytest.raises(ValueError, match="primary"):
            ConfigLoader.load_from_json(path)
    finally:
        os.unlink(path)


def test_invalid_primary_nonexistent_provider():
    """primary referencing unknown provider raises ValueError."""
    cfg = _valid_two_provider_config()
    cfg["primary"] = "nonexist/model-a"
    path = _write_temp_config(cfg)
    try:
        with pytest.raises(ValueError, match="unknown provider"):
            ConfigLoader.load_from_json(path)
    finally:
        os.unlink(path)


def test_invalid_primary_nonexistent_model():
    """primary referencing unknown model raises ValueError."""
    cfg = _valid_two_provider_config()
    cfg["primary"] = "aliyun/nonexist"
    path = _write_temp_config(cfg)
    try:
        with pytest.raises(ValueError, match="unknown model"):
            ConfigLoader.load_from_json(path)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 7-8. Fallback validation
# ---------------------------------------------------------------------------

def test_invalid_fallback_reference():
    """Fallback referencing unknown provider/model raises ValueError."""
    cfg = _valid_two_provider_config()
    cfg["fallback"] = ["nonexist/model"]
    path = _write_temp_config(cfg)
    try:
        with pytest.raises(ValueError, match="unknown provider"):
            ConfigLoader.load_from_json(path)
    finally:
        os.unlink(path)


def test_empty_fallback_valid():
    """Config without 'fallback' key loads successfully with fallback=[]."""
    cfg = _valid_two_provider_config()
    del cfg["fallback"]
    path = _write_temp_config(cfg)
    try:
        config = ConfigLoader.load_from_json(path)
        assert config.fallback == []
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 9. Minimal config
# ---------------------------------------------------------------------------

def test_minimal_config():
    """One provider, one model, no fallback loads successfully."""
    cfg = {
        "providers": {
            "single": {
                "api_key": "key",
                "base_url": "https://api.test.com/v1",
                "models": {"model-a": {}},
            }
        },
        "primary": "single/model-a",
    }
    path = _write_temp_config(cfg)
    try:
        config = ConfigLoader.load_from_json(path)
        assert len(config.providers) == 1
        assert config.primary == "single/model-a"
        assert config.fallback == []
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 10-12. Reserved keys in model params
# ---------------------------------------------------------------------------

def test_reserved_key_in_model_params():
    """Model params with reserved key 'model' raises ValueError."""
    cfg = _valid_two_provider_config()
    cfg["providers"]["aliyun"]["models"]["deepseek-v4-pro"] = {"model": "override"}
    path = _write_temp_config(cfg)
    try:
        with pytest.raises(ValueError, match="reserved"):
            ConfigLoader.load_from_json(path)
    finally:
        os.unlink(path)


def test_reserved_key_messages():
    """Model params with reserved key 'messages' raises ValueError."""
    cfg = _valid_two_provider_config()
    cfg["providers"]["aliyun"]["models"]["deepseek-v4-pro"] = {
        "messages": [{"role": "user"}]
    }
    path = _write_temp_config(cfg)
    try:
        with pytest.raises(ValueError, match="reserved"):
            ConfigLoader.load_from_json(path)
    finally:
        os.unlink(path)


def test_reserved_key_tools():
    """Model params with reserved key 'tools' raises ValueError."""
    cfg = _valid_two_provider_config()
    cfg["providers"]["aliyun"]["models"]["deepseek-v4-pro"] = {"tools": []}
    path = _write_temp_config(cfg)
    try:
        with pytest.raises(ValueError, match="reserved"):
            ConfigLoader.load_from_json(path)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 13. Valid model params
# ---------------------------------------------------------------------------

def test_valid_model_params():
    """Model params with temperature and reasoning_effort load successfully."""
    cfg = _valid_two_provider_config()
    cfg["providers"]["aliyun"]["models"]["deepseek-v4-pro"] = {
        "temperature": 0.5,
        "reasoning_effort": "high",
    }
    path = _write_temp_config(cfg)
    try:
        config = ConfigLoader.load_from_json(path)
        model_params = config.providers["aliyun"].models["deepseek-v4-pro"]
        assert model_params["temperature"] == 0.5
        assert model_params["reasoning_effort"] == "high"
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 14-16. resolve_model_ref
# ---------------------------------------------------------------------------

def test_resolve_model_ref_basic():
    """Basic 'provider/model' splits correctly."""
    provider, model = resolve_model_ref("aliyun/deepseek-v4-pro")
    assert provider == "aliyun"
    assert model == "deepseek-v4-pro"


def test_resolve_model_ref_with_dots():
    """Model names containing dots parse correctly."""
    provider, model = resolve_model_ref("aliyun/claude-3.5-sonnet")
    assert provider == "aliyun"
    assert model == "claude-3.5-sonnet"


def test_resolve_model_ref_no_slash():
    """Reference without '/' raises ValueError."""
    with pytest.raises(ValueError, match="provider/model"):
        resolve_model_ref("invalid")


# ---------------------------------------------------------------------------
# 17. RPM limit default
# ---------------------------------------------------------------------------

def test_rpm_limit_default():
    """Provider without rpm_limit gets default value 100."""
    cfg = _valid_two_provider_config()
    for prov in cfg["providers"].values():
        prov.pop("rpm_limit", None)
    path = _write_temp_config(cfg)
    try:
        config = ConfigLoader.load_from_json(path)
        for prov_conf in config.providers.values():
            assert prov_conf.rpm_limit == 100.0
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 18. engine.json.example loads
# ---------------------------------------------------------------------------

def test_engine_json_example_loads():
    """engine.json.example loads without error."""
    example_path = os.path.join(os.path.dirname(__file__), "..", "engine.json.example")
    example_path = os.path.abspath(example_path)
    config = ConfigLoader.load_from_json(example_path)
    assert isinstance(config, Config)
    assert "primary" in config.providers
    assert "fallback" in config.providers
    assert config.primary == "primary/gpt-4"
