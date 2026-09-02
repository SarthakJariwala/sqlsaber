"""Tests for settings and configuration management."""

import json
import os
import platform
import stat
from unittest.mock import Mock, patch

import pytest

from sqlsaber.config.settings import Config, ModelConfigManager, ThinkingLevel


class TestModelConfigManager:
    """Test the ModelConfigManager class."""

    @pytest.fixture
    def model_manager(self, temp_dir, monkeypatch):
        """Create a ModelConfigManager with temp directory."""

        config_dir = temp_dir / "config"
        monkeypatch.setattr(
            "platformdirs.user_config_dir", lambda *args, **kwargs: str(config_dir)
        )
        return ModelConfigManager()

    def test_initialization(self, model_manager):
        """Test manager initialization creates config directory."""

        assert model_manager.config_dir.exists()
        assert model_manager.config_file.name == "model_config.json"

    @pytest.mark.skipif(platform.system() == "Windows", reason="Unix permissions test")
    def test_secure_permissions_unix(self, model_manager):
        """Test secure permissions are set on Unix systems."""

        dir_stat = os.stat(model_manager.config_dir)
        dir_perms = stat.S_IMODE(dir_stat.st_mode)
        assert dir_perms == 0o700

    def test_default_model(self, model_manager):
        """Test default model is returned when no config exists."""

        model = model_manager.get_model()
        assert model == ModelConfigManager.DEFAULT_MODEL
        assert model == "openai:gpt-5.6-sol"

    def test_cli_reset_target_is_the_config_default(self):
        from sqlsaber.cli.models import ModelManager

        assert ModelManager.DEFAULT_MODEL == ModelConfigManager.DEFAULT_MODEL
        assert ModelManager.DEFAULT_MODEL == "openai:gpt-5.6-sol"

    def test_set_and_get_model(self, model_manager):
        """Test setting and retrieving a model."""

        test_model = "anthropic:claude-3-opus-20240229"
        model_manager.set_model(test_model)

        assert model_manager.get_model() == test_model

        new_manager = ModelConfigManager()
        new_manager.config_dir = model_manager.config_dir
        new_manager.config_file = model_manager.config_file
        assert new_manager.get_model() == test_model

    def test_get_subagent_model_unset(self, model_manager):
        """Test getting a subagent model when unset."""

        assert model_manager.get_subagent_model("handoff") is None

    def test_set_and_clear_subagent_model(self, model_manager):
        """Test setting and clearing subagent model overrides."""

        handoff_model = "openai:gpt-4o-mini"
        viz_model = "anthropic:claude-sonnet-4-5-20250929"

        model_manager.set_subagent_model("handoff", handoff_model)
        model_manager.set_subagent_model("viz", viz_model)

        assert model_manager.get_subagent_model("handoff") == handoff_model
        assert model_manager.get_subagent_model("viz") == viz_model

        config = model_manager._load_config()
        assert config["subagents"] == {"handoff": handoff_model, "viz": viz_model}

        model_manager.set_subagent_model("handoff", None)
        config = model_manager._load_config()
        assert config["subagents"] == {"viz": viz_model}

        model_manager.set_subagent_model("viz", None)
        config = model_manager._load_config()
        assert "subagents" not in config

    def test_config_file_format(self, model_manager):
        """Test the config file is properly formatted (v2 format)."""

        test_model = "anthropic:claude-sonnet-4"
        model_manager.set_model(test_model)

        with open(model_manager.config_file, "r") as f:
            config = json.load(f)

        assert config == {
            "version": 2,
            "model": test_model,
            "thinking": {
                "enabled": True,
                "level": "medium",
            },
        }
        assert model_manager.config_file.read_text().strip().endswith("}")

    def test_corrupted_config_file(self, model_manager):
        """Test handling of corrupted config file."""

        model_manager.config_file.parent.mkdir(parents=True, exist_ok=True)
        model_manager.config_file.write_text("invalid json{")

        assert model_manager.get_model() == ModelConfigManager.DEFAULT_MODEL

    def test_missing_model_in_config(self, model_manager):
        """Test handling of config file without model key."""

        model_manager.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(model_manager.config_file, "w") as f:
            json.dump({"other_key": "value"}, f)

        assert model_manager.get_model() == ModelConfigManager.DEFAULT_MODEL


class TestConfig:
    """Test the Config class."""

    @pytest.fixture
    def config(self, temp_dir, monkeypatch):
        """Create a Config instance with mocked dependencies."""

        config_dir = temp_dir / "config"
        monkeypatch.setattr(
            "platformdirs.user_config_dir", lambda *args, **kwargs: str(config_dir)
        )

        with patch("sqlsaber.config.settings.APIKeyManager") as mock_api_key_manager:
            mock_manager = Mock()
            mock_api_key_manager.return_value = mock_manager
            mock_manager.get_api_key.return_value = "test-api-key"

            config = Config()
            config.auth._api_key_manager = mock_manager
            return config

    def test_initialization(self, config):
        """Test Config initialization."""

        assert config.model_name == ModelConfigManager.DEFAULT_MODEL
        assert config.api_key == "test-api-key"

    def test_get_api_key_anthropic(self, config):
        """Test API key retrieval for Anthropic models."""

        config.model_name = "anthropic:claude-3-opus"
        api_key = config.api_key

        config.auth._api_key_manager.get_api_key.assert_called_with("anthropic")
        assert api_key == "test-api-key"

    def test_set_model(self, config):
        """Test setting a new model updates configuration."""

        new_model = "openai:gpt-4-turbo"
        config.auth._api_key_manager.get_api_key.return_value = "new-api-key"

        config.set_model(new_model)

        assert config.model_name == new_model
        assert config.model._manager.get_model() == new_model

    def test_validate_success(self, config):
        """Test successful validation when API key exists."""

        config.auth._api_key_manager.get_api_key.return_value = "valid-key"
        config.validate()

    def test_validate_missing_anthropic_key(self, config):
        """Test validation error for missing Anthropic API key."""

        config.model_name = "anthropic:claude-3"
        config.auth._api_key_manager.get_api_key.return_value = None

        with pytest.raises(ValueError, match="Anthropic API key not found"):
            config.validate()

    def test_in_memory_config_avoids_platformdirs(self, monkeypatch):
        """Config.in_memory should not touch filesystem-backed platformdirs paths."""

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        def _fail(*args, **kwargs):
            _ = args, kwargs
            raise AssertionError("platformdirs.user_config_dir should not be called")

        monkeypatch.setattr("platformdirs.user_config_dir", _fail)

        config = Config.in_memory(
            model_name="openai:gpt-5-mini",
            thinking_enabled=True,
            thinking_level=ThinkingLevel.HIGH,
            api_keys={"openai": "test-api-key"},
        )

        assert config.model_name == "openai:gpt-5-mini"
        assert config.model.thinking_enabled is True
        assert config.model.thinking_level == ThinkingLevel.HIGH
        assert config.api_key == "test-api-key"

    def test_in_memory_config_accepts_notebook_subagent(self):
        config = Config.in_memory(
            model_name="openai:gpt-5-mini",
            subagent_models={
                "notebook": "anthropic:claude-sonnet",
                "unknown": "openai:gpt-unknown",
            },
        )

        assert config.model.get_subagent_models() == {
            "notebook": "anthropic:claude-sonnet"
        }

    def test_in_memory_config_validate_errors_without_api_key(self, monkeypatch):
        """In-memory config should fail validation when no API key is provided."""

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = Config.in_memory(model_name="openai:gpt-5-mini")
        with pytest.raises(ValueError, match="Openai API key not found"):
            config.validate()

    def test_in_memory_thinking_stays_false_by_default(self):
        config = Config.in_memory()
        assert config.model.thinking_enabled is False


class TestThinkingLevel:
    """Test the ThinkingLevel enum."""

    def test_all_levels_exist(self):
        """Test that all expected thinking levels exist."""

        expected = {"minimal", "low", "medium", "high", "maximum"}
        actual = {level.value for level in ThinkingLevel}
        assert actual == expected

    def test_from_string_valid(self):
        """Test converting valid strings to ThinkingLevel."""

        assert ThinkingLevel.from_string("minimal") == ThinkingLevel.MINIMAL
        assert ThinkingLevel.from_string("low") == ThinkingLevel.LOW
        assert ThinkingLevel.from_string("medium") == ThinkingLevel.MEDIUM
        assert ThinkingLevel.from_string("high") == ThinkingLevel.HIGH
        assert ThinkingLevel.from_string("maximum") == ThinkingLevel.MAXIMUM

    def test_from_string_case_insensitive(self):
        """Test that from_string is case insensitive."""

        assert ThinkingLevel.from_string("MEDIUM") == ThinkingLevel.MEDIUM
        assert ThinkingLevel.from_string("High") == ThinkingLevel.HIGH

    def test_from_string_invalid_defaults_to_medium(self):
        """Test that invalid strings default to MEDIUM."""

        assert ThinkingLevel.from_string("invalid") == ThinkingLevel.MEDIUM
        assert ThinkingLevel.from_string("") == ThinkingLevel.MEDIUM
        assert ThinkingLevel.from_string("super_high") == ThinkingLevel.MEDIUM
        assert ThinkingLevel.from_string("off") == ThinkingLevel.MEDIUM


class TestModelConfigManagerV2:
    """Test v2 config schema and migration."""

    @pytest.fixture
    def model_manager(self, temp_dir, monkeypatch):
        """Create a ModelConfigManager with temp directory."""

        config_dir = temp_dir / "config"
        monkeypatch.setattr(
            "platformdirs.user_config_dir", lambda *args, **kwargs: str(config_dir)
        )
        return ModelConfigManager()

    def test_default_config_is_v2(self, model_manager):
        """Test that default config is v2 format."""

        config = model_manager._load_config()
        assert config.get("version") == 2
        assert "thinking" in config
        assert config["thinking"]["enabled"] is True
        assert config["thinking"]["level"] == ThinkingLevel.MEDIUM.value

    def test_migrate_v1_to_v2_thinking_disabled(self, model_manager):
        """Test migration of v1 config with thinking disabled."""

        model_manager.config_file.parent.mkdir(parents=True, exist_ok=True)
        v1_config = {
            "model": "anthropic:claude-3",
            "thinking_enabled": False,
        }
        with open(model_manager.config_file, "w") as f:
            json.dump(v1_config, f)

        config = model_manager._load_config()
        assert config["version"] == 2
        assert config["model"] == "anthropic:claude-3"
        assert config["thinking"]["enabled"] is False
        assert config["thinking"]["level"] == ThinkingLevel.MEDIUM.value

    def test_migrate_v1_to_v2_thinking_enabled(self, model_manager):
        """Test migration of v1 config with thinking enabled."""

        model_manager.config_file.parent.mkdir(parents=True, exist_ok=True)
        v1_config = {
            "model": "anthropic:claude-sonnet-4",
            "thinking_enabled": True,
        }
        with open(model_manager.config_file, "w") as f:
            json.dump(v1_config, f)

        config = model_manager._load_config()
        assert config["version"] == 2
        assert config["model"] == "anthropic:claude-sonnet-4"
        assert config["thinking"]["enabled"] is True
        assert config["thinking"]["level"] == ThinkingLevel.MEDIUM.value

    def test_migration_persists_to_file(self, model_manager):
        """Test that migration saves the v2 config to file."""

        model_manager.config_file.parent.mkdir(parents=True, exist_ok=True)
        v1_config = {
            "model": "openai:gpt-4",
            "thinking_enabled": True,
        }
        with open(model_manager.config_file, "w") as f:
            json.dump(v1_config, f)

        model_manager._load_config()

        with open(model_manager.config_file, "r") as f:
            saved_config = json.load(f)

        assert saved_config["version"] == 2
        assert "thinking" in saved_config
        assert "thinking_enabled" not in saved_config

    def test_get_and_set_thinking_level(self, model_manager):
        """Test getting and setting thinking level."""

        assert model_manager.get_thinking_level() == ThinkingLevel.MEDIUM

        model_manager.set_thinking_level(ThinkingLevel.HIGH)
        assert model_manager.get_thinking_level() == ThinkingLevel.HIGH

        model_manager.set_thinking_level(ThinkingLevel.MINIMAL)
        assert model_manager.get_thinking_level() == ThinkingLevel.MINIMAL

    def test_set_thinking_both_enabled_and_level(self, model_manager):
        """Test setting both thinking enabled and level atomically."""

        model_manager.set_thinking(enabled=True, level=ThinkingLevel.HIGH)

        assert model_manager.get_thinking_enabled() is True
        assert model_manager.get_thinking_level() == ThinkingLevel.HIGH

        model_manager.set_thinking(enabled=False, level=ThinkingLevel.LOW)

        assert model_manager.get_thinking_enabled() is False
        assert model_manager.get_thinking_level() == ThinkingLevel.LOW

    def test_v2_config_file_format(self, model_manager):
        """Test the v2 config file format."""

        model_manager.set_model("google:gemini-pro")
        model_manager.set_thinking(enabled=True, level=ThinkingLevel.MAXIMUM)

        with open(model_manager.config_file, "r") as f:
            config = json.load(f)

        assert config == {
            "version": 2,
            "model": "google:gemini-pro",
            "thinking": {
                "enabled": True,
                "level": "maximum",
            },
        }

    def test_missing_config_thinking_defaults_without_writing_file(self, model_manager):
        assert not model_manager.config_file.exists()
        assert model_manager.get_thinking_enabled() is True
        assert model_manager.get_thinking_level() is ThinkingLevel.MEDIUM
        assert not model_manager.config_file.exists()

    def test_incomplete_v2_missing_thinking_fills_enabled_true_without_write(
        self, model_manager
    ):
        payload = {"version": 2, "model": "openai:gpt-4"}
        text = json.dumps(payload)
        model_manager.config_file.write_text(text)

        assert model_manager.get_thinking_enabled() is True
        assert model_manager.get_thinking_level() is ThinkingLevel.MEDIUM
        assert model_manager.config_file.read_text() == text

    def test_incomplete_v2_missing_enabled_fills_true_without_write(
        self, model_manager
    ):
        payload = {
            "version": 2,
            "model": "openai:gpt-4",
            "thinking": {"level": "high"},
        }
        text = json.dumps(payload)
        model_manager.config_file.write_text(text)

        assert model_manager.get_thinking_enabled() is True
        assert model_manager.get_thinking_level() is ThinkingLevel.HIGH
        assert model_manager.config_file.read_text() == text

    def test_complete_v2_enabled_false_stays_false_and_file_unchanged(
        self, model_manager
    ):
        payload = {
            "version": 2,
            "model": "openai:gpt-4",
            "thinking": {"enabled": False, "level": "low"},
        }
        text = json.dumps(payload, indent=2)
        model_manager.config_file.write_text(text)

        assert model_manager.get_thinking_enabled() is False
        assert model_manager.get_thinking_level() is ThinkingLevel.LOW
        assert model_manager.config_file.read_text() == text

    def test_set_model_then_reset_preserves_thinking_false(self, model_manager):
        from sqlsaber.cli.models import ModelManager

        payload = {
            "version": 2,
            "model": "anthropic:claude-3",
            "thinking": {"enabled": False, "level": "high"},
        }
        model_manager.config_file.write_text(json.dumps(payload, indent=2))

        cli_manager = ModelManager()
        assert cli_manager.set_model("google:gemini-pro") is True
        assert model_manager.get_thinking_enabled() is False
        assert cli_manager.reset_model() is True
        assert model_manager.get_model() == ModelConfigManager.DEFAULT_MODEL
        assert model_manager.get_thinking_enabled() is False
        assert model_manager.get_thinking_level() is ThinkingLevel.HIGH

    def test_v1_missing_thinking_enabled_migrates_to_false(self, model_manager):
        v1_config = {"model": "anthropic:claude-3"}
        model_manager.config_file.write_text(json.dumps(v1_config))

        config = model_manager._load_config()
        assert config["thinking"]["enabled"] is False
        saved = json.loads(model_manager.config_file.read_text())
        assert saved["thinking"]["enabled"] is False

    def test_corrupt_config_thinking_defaults_without_repair(self, model_manager):
        model_manager.config_file.write_text("invalid json{")

        assert model_manager.get_thinking_enabled() is True
        assert model_manager.get_thinking_level() is ThinkingLevel.MEDIUM
        assert model_manager.config_file.read_text() == "invalid json{"

    def test_level_off_forces_enabled_false(self, model_manager):
        payload = {
            "version": 2,
            "model": "openai:gpt-4",
            "thinking": {"enabled": True, "level": "off"},
        }
        model_manager.config_file.write_text(json.dumps(payload))

        assert model_manager.get_thinking_enabled() is False
        assert model_manager.get_thinking_level() is ThinkingLevel.MEDIUM
