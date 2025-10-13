"""Tests for configuration module.

This module tests configuration loading, environment variables,
and configuration validation.
"""

import os
from unittest.mock import patch

from nav2_mcp_server.config import get_config


class TestConfigLoading:
    """Tests for configuration loading functionality."""

    @patch.dict(os.environ, {}, clear=True)
    def test_get_config_defaults(self):
        """Test configuration loading with default values.

        Verifies that the configuration loads successfully
        with default values when no environment variables are set.
        """
        config = get_config()

        assert config is not None
        assert hasattr(config, 'server')
        assert hasattr(config, 'navigation')
        assert hasattr(config, 'transforms')
        assert hasattr(config, 'logging')

    def test_get_config_server_section(self):
        """Test server configuration section.

        Verifies that server configuration is properly loaded
        with expected attributes.
        """
        config = get_config()

        assert hasattr(config.server, 'server_name')
        assert hasattr(config.server, 'pose_uri')
        assert isinstance(config.server.server_name, str)
        assert isinstance(config.server.pose_uri, str)

    def test_get_config_navigation_section(self):
        """Test navigation configuration section.

        Verifies that navigation configuration contains
        expected parameters.
        """
        config = get_config()

        assert hasattr(config.navigation, 'default_frame_id')
        assert hasattr(config.navigation, 'default_timeout')
        assert isinstance(config.navigation.default_frame_id, str)
        assert isinstance(config.navigation.default_timeout, (int, float))

    def test_get_config_transforms_section(self):
        """Test transforms configuration section.

        Verifies that transform configuration contains
        expected frame parameters.
        """
        config = get_config()

        assert hasattr(config.transforms, 'map_frame')
        assert hasattr(config.transforms, 'base_frame')
        assert isinstance(config.transforms.map_frame, str)
        assert isinstance(config.transforms.base_frame, str)

    def test_get_config_logging_section(self):
        """Test logging configuration section.

        Verifies that logging configuration contains
        expected parameters.
        """
        config = get_config()

        assert hasattr(config.logging, 'log_level')
        assert hasattr(config.logging, 'log_format')
        assert isinstance(config.logging.log_level, str)
        assert isinstance(config.logging.log_format, str)


class TestEnvironmentVariables:
    """Tests for environment variable handling."""

    @patch.dict(os.environ, {'NAV2_MCP_SERVER_NAME': 'TestServer'})
    def test_server_name_environment_variable(self):
        """Test server name from environment variable.

        Verifies that server name can be configured via
        environment variable.
        """
        config = get_config()

        assert config.server.server_name == 'TestServer'

    @patch.dict(os.environ, {'NAV2_MCP_LOG_LEVEL': 'DEBUG'})
    def test_log_level_environment_variable(self):
        """Test log level from environment variable.

        Verifies that log level can be configured via
        environment variable.
        """
        config = get_config()

        assert config.logging.log_level == 'DEBUG'

    @patch.dict(os.environ, {'NAV2_MCP_MAP_FRAME': 'world'})
    def test_map_frame_environment_variable(self):
        """Test map frame from environment variable.

        Verifies that map frame can be configured via
        environment variable.
        """
        config = get_config()

        assert config.transforms.map_frame == 'world'

    @patch.dict(os.environ, {'NAV2_MCP_BASE_FRAME': 'robot_base'})
    def test_base_frame_environment_variable(self):
        """Test base frame from environment variable.

        Verifies that base frame can be configured via
        environment variable.
        """
        config = get_config()

        assert config.transforms.base_frame == 'robot_base'

    @patch.dict(
        os.environ,
        {
            'NAV2_MCP_SERVER_NAME': 'EnvServer',
            'NAV2_MCP_LOG_LEVEL': 'WARNING',
            'NAV2_MCP_MAP_FRAME': 'env_map',
            'NAV2_MCP_BASE_FRAME': 'env_base'
        }
    )
    def test_multiple_environment_variables(self):
        """Test multiple environment variables.

        Verifies that multiple environment variables
        can be set simultaneously.
        """
        config = get_config()

        assert config.server.server_name == 'EnvServer'
        assert config.logging.log_level == 'WARNING'
        assert config.transforms.map_frame == 'env_map'
        assert config.transforms.base_frame == 'env_base'


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_config_types(self):
        """Test configuration parameter types.

        Verifies that configuration parameters have
        the expected data types.
        """
        config = get_config()

        # String parameters
        assert isinstance(config.server.server_name, str)
        assert isinstance(config.server.pose_uri, str)
        assert isinstance(config.navigation.default_frame_id, str)
        assert isinstance(config.transforms.map_frame, str)
        assert isinstance(config.transforms.base_frame, str)
        assert isinstance(config.logging.log_level, str)
        assert isinstance(config.logging.log_format, str)

        # Numeric parameters
        assert isinstance(config.navigation.default_timeout, (int, float))

    def test_config_non_empty_strings(self):
        """Test that string configuration values are non-empty.

        Verifies that required string parameters
        are not empty.
        """
        config = get_config()

        assert len(config.server.server_name) > 0
        assert len(config.server.pose_uri) > 0
        assert len(config.navigation.default_frame_id) > 0
        assert len(config.transforms.map_frame) > 0
        assert len(config.transforms.base_frame) > 0
        assert len(config.logging.log_level) > 0
        assert len(config.logging.log_format) > 0

    def test_config_timeout_positive(self):
        """Test that timeout values are positive.

        Verifies that timeout configuration values
        are positive numbers.
        """
        config = get_config()

        assert config.navigation.default_timeout > 0

    def test_config_valid_log_levels(self):
        """Test valid log level values.

        Verifies that log level is set to a valid logging level.
        """
        config = get_config()

        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        assert config.logging.log_level in valid_levels


class TestConfigurationConsistency:
    """Tests for configuration consistency and relationships."""

    def test_config_singleton_behavior(self):
        """Test configuration singleton behavior.

        Verifies that get_config returns the same
        configuration instance on repeated calls.
        """
        config1 = get_config()
        config2 = get_config()

        # Should return the same configuration values
        assert config1.server.server_name == config2.server.server_name
        assert config1.logging.log_level == config2.logging.log_level
        assert config1.transforms.map_frame == config2.transforms.map_frame

    def test_config_frame_names_different(self):
        """Test that map and base frame names are different.

        Verifies that map frame and base frame have
        different names to avoid transform issues.
        """
        config = get_config()

        assert config.transforms.map_frame != config.transforms.base_frame

    def test_config_uri_format(self):
        """Test URI format validation.

        Verifies that URI values follow expected format patterns.
        """
        config = get_config()

        # Pose URI should be a valid URI format
        assert '://' in config.server.pose_uri or config.server.pose_uri.startswith('/')


class TestConfigurationErrors:
    """Tests for configuration error handling."""

    @patch.dict(os.environ, {'NAV2_MCP_LOG_LEVEL': 'INVALID_LEVEL'})
    def test_invalid_log_level_handling(self):
        """Test handling of invalid log level values.

        Verifies that invalid log levels are handled
        gracefully with fallback to default.
        """
        config = get_config()

        # Should fallback to a valid log level
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        assert config.logging.log_level in valid_levels

    @patch.dict(os.environ, {'NAV2_MCP_DEFAULT_TIMEOUT': 'not_a_number'})
    def test_invalid_timeout_handling(self):
        """Test handling of invalid timeout values.

        Verifies that invalid timeout values are handled
        gracefully with fallback to default.
        """
        config = get_config()

        # Should fallback to a valid positive number
        assert isinstance(config.navigation.default_timeout, (int, float))
        assert config.navigation.default_timeout > 0

    def test_config_loading_robustness(self):
        """Test configuration loading robustness.

        Verifies that configuration loading is robust
        against various error conditions.
        """
        # Should not raise exceptions
        config = get_config()
        assert config is not None

        # Should have all required sections
        assert hasattr(config, 'server')
        assert hasattr(config, 'navigation')
        assert hasattr(config, 'transforms')
        assert hasattr(config, 'logging')


class TestConfigurationDefaults:
    """Tests for configuration default values."""

    @patch.dict(os.environ, {}, clear=True)
    def test_default_server_name(self):
        """Test default server name value.

        Verifies that the default server name is appropriate.
        """
        config = get_config()

        # Should contain "Nav2" or "MCP" in the name
        server_name = config.server.server_name.lower()
        assert 'nav2' in server_name or 'mcp' in server_name

    @patch.dict(os.environ, {}, clear=True)
    def test_default_frame_names(self):
        """Test default frame name values.

        Verifies that default frame names follow ROS conventions.
        """
        config = get_config()

        # Common ROS frame names
        assert config.transforms.map_frame in ['map', 'world', 'odom']
        assert config.transforms.base_frame in ['base_link', 'base_footprint', 'robot_base']

    @patch.dict(os.environ, {}, clear=True)
    def test_default_logging_configuration(self):
        """Test default logging configuration.

        Verifies that default logging settings are reasonable.
        """
        config = get_config()

        # Should have reasonable default log level
        assert config.logging.log_level in ['INFO', 'DEBUG', 'WARNING']

        # Log format should contain basic elements
        log_format = config.logging.log_format
        assert 'levelname' in log_format.lower() or 'level' in log_format.lower()
        assert 'message' in log_format.lower()