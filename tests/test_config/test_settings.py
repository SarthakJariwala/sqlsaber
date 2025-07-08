"""Tests for settings and configuration management."""

import json
import os
import platform
import stat
from unittest.mock import Mock, patch

import pytest

from sqlsaber.config.settings import Config, ModelConfigManager


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
        # Check directory permissions
        dir_stat = os.stat(model_manager.config_dir)
        dir_perms = stat.S_IMODE(dir_stat.st_mode)
        assert dir_perms == 0o700

    def test_default_model(self, model_manager):
        """Test default model is returned when no config exists."""
        model = model_manager.get_model()
        assert model == ModelConfigManager.DEFAULT_MODEL

    def test_set_and_get_model(self, model_manager):
        """Test setting and retrieving a model."""
        test_model = "anthropic:claude-3-opus-20240229"
        model_manager.set_model(test_model)

        # Verify it was saved
        assert model_manager.get_model() == test_model

        # Verify it persists (create new instance)
        new_manager = ModelConfigManager()
        new_manager.config_dir = model_manager.config_dir
        new_manager.config_file = model_manager.config_file
        assert new_manager.get_model() == test_model

    def test_config_file_format(self, model_manager):
        """Test the config file is properly formatted."""
        test_model = "anthropic:claude-sonnet-4"
        model_manager.set_model(test_model)

        # Read the config file directly
        with open(model_manager.config_file, "r") as f:
            config = json.load(f)

        assert config == {"model": test_model}
        assert model_manager.config_file.read_text().strip().endswith("}")

    def test_corrupted_config_file(self, model_manager):
        """Test handling of corrupted config file."""
        # Write invalid JSON
        model_manager.config_file.parent.mkdir(parents=True, exist_ok=True)
        model_manager.config_file.write_text("invalid json{")

        # Should return default model
        assert model_manager.get_model() == ModelConfigManager.DEFAULT_MODEL

    def test_missing_model_in_config(self, model_manager):
        """Test handling of config file without model key."""
        # Write config without model key
        model_manager.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(model_manager.config_file, "w") as f:
            json.dump({"other_key": "value"}, f)

        # Should return default model
        assert model_manager.get_model() == ModelConfigManager.DEFAULT_MODEL


class TestConfig:
    """Test the Config class."""

    @pytest.fixture
    def config(self, temp_dir, monkeypatch):
        """Create a Config instance with mocked dependencies."""
        # Mock platformdirs
        config_dir = temp_dir / "config"
        monkeypatch.setattr(
            "platformdirs.user_config_dir", lambda *args, **kwargs: str(config_dir)
        )

        # Mock API key manager
        with patch("sqlsaber.config.settings.APIKeyManager") as mock_api_key_manager:
            mock_manager = Mock()
            mock_api_key_manager.return_value = mock_manager
            mock_manager.get_api_key.return_value = "test-api-key"

            config = Config()
            config._mock_api_key_manager = mock_manager
            return config

    def test_initialization(self, config):
        """Test Config initialization."""
        assert config.model_name == ModelConfigManager.DEFAULT_MODEL
        assert config.api_key == "test-api-key"

    def test_get_api_key_anthropic(self, config):
        """Test API key retrieval for Anthropic models."""
        config.model_name = "anthropic:claude-3-opus"
        api_key = config._get_api_key()

        config._mock_api_key_manager.get_api_key.assert_called_with("anthropic")
        assert api_key == "test-api-key"

    def test_set_model(self, config):
        """Test setting a new model updates configuration."""
        new_model = "openai:gpt-4-turbo"
        config._mock_api_key_manager.get_api_key.return_value = "new-api-key"

        config.set_model(new_model)

        assert config.model_name == new_model
        assert config.model_config_manager.get_model() == new_model

    def test_validate_success(self, config):
        """Test successful validation when API key exists."""
        config.api_key = "valid-key"
        config.validate()  # Should not raise

    def test_validate_missing_anthropic_key(self, config):
        """Test validation error for missing Anthropic API key."""
        config.model_name = "anthropic:claude-3"
        config.api_key = None

        with pytest.raises(ValueError, match="Anthropic API key not found"):
            config.validate()
