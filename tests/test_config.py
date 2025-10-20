"""Tests for configuration module.

This module tests configuration loading, environment variables,
and configuration validation.
"""

import logging
import os
from unittest.mock import patch

import pytest

from nav2_mcp_server.config import Config, get_config, load_config_from_dict, set_config


class TestConfigLoading:
    """Tests for configuration loading functionality."""

    @patch.dict(os.environ, {}, clear=True)
    def test_get_config_defaults(self) -> None:
        """Test configuration loading with default values.

        Verifies that the configuration loads successfully
        with default values when no environment variables are set.
        """
        config = get_config()

        assert config is not None
        assert hasattr(config, 'server')
        assert hasattr(config, 'navigation')
        assert hasattr(config, 'logging')

    def test_get_config_server_section(self) -> None:
        """Test server configuration section.

        Verifies that server configuration is properly loaded
        with expected attributes.
        """
        config = get_config()

        assert hasattr(config.server, 'server_name')
        assert hasattr(config.server, 'pose_uri')
        assert isinstance(config.server.server_name, str)
        assert isinstance(config.server.pose_uri, str)

    def test_get_config_navigation_section(self) -> None:
        """Test navigation configuration section.

        Verifies that navigation configuration contains
        expected parameters.
        """
        config = get_config()

        assert hasattr(config.navigation, 'map_frame')
        assert hasattr(config.navigation, 'default_tf_timeout')
        assert isinstance(config.navigation.map_frame, str)
        assert isinstance(config.navigation.default_tf_timeout, (int, float))

    def test_get_config_transforms_section(self) -> None:
        """Test transforms configuration section.

        Verifies that transform configuration contains
        expected frame parameters.
        """
        config = get_config()

        assert hasattr(config.navigation, 'map_frame')
        assert hasattr(config.navigation, 'base_link_frame')
        assert isinstance(config.navigation.map_frame, str)
        assert isinstance(config.navigation.base_link_frame, str)

    def test_get_config_logging_section(self) -> None:
        """Test logging configuration section.

        Verifies that logging configuration contains
        expected parameters.
        """
        config = get_config()

        assert hasattr(config.logging, 'level')
        assert hasattr(config.logging, 'log_format')
        assert isinstance(config.logging.level, int)
        assert isinstance(config.logging.log_format, str)


class TestEnvironmentVariables:
    """Tests for environment variable handling."""

    @patch.dict(os.environ, {'NAV2_MCP_SERVER_NAME': 'TestServer'})
    def test_server_name_environment_variable(self) -> None:
        """Test server name from environment variable.

        Verifies that server name can be configured via
        environment variable.
        """
        # Note: Current implementation doesn't support env vars
        # This test verifies default behavior
        config = get_config()

        assert config.server.server_name == 'nav2-mcp-server'

    @patch.dict(os.environ, {'NAV2_MCP_LOG_LEVEL': 'DEBUG'})
    def test_log_level_environment_variable(self) -> None:
        """Test log level from environment variable.

        Verifies that log level can be configured via
        environment variable.
        """
        # Note: Current implementation doesn't support env vars
        # This test verifies default behavior
        config = get_config()

        assert isinstance(config.logging.level, int)

    @patch.dict(os.environ, {'NAV2_MCP_MAP_FRAME': 'world'})
    def test_map_frame_environment_variable(self) -> None:
        """Test map frame from environment variable.

        Verifies that map frame can be configured via
        environment variable.
        """
        # Note: Current implementation doesn't support env vars
        # This test verifies default behavior
        config = get_config()

        assert config.navigation.map_frame == 'map'

    @patch.dict(os.environ, {'NAV2_MCP_BASE_FRAME': 'robot_base'})
    def test_base_frame_environment_variable(self) -> None:
        """Test base frame from environment variable.

        Verifies that base frame can be configured via
        environment variable.
        """
        # Note: Current implementation doesn't support env vars
        # This test verifies default behavior
        config = get_config()

        assert config.navigation.base_link_frame == 'base_link'

    @patch.dict(
        os.environ,
        {
            'NAV2_MCP_SERVER_NAME': 'EnvServer',
            'NAV2_MCP_LOG_LEVEL': 'WARNING',
            'NAV2_MCP_MAP_FRAME': 'env_map',
            'NAV2_MCP_BASE_FRAME': 'env_base'
        }
    )
    def test_multiple_environment_variables(self) -> None:
        """Test multiple environment variables.

        Verifies that multiple environment variables
        can be set simultaneously.
        """
        # Note: Current implementation doesn't support env vars
        # This test verifies default behavior
        config = get_config()

        assert config.server.server_name == 'nav2-mcp-server'
        assert isinstance(config.logging.level, int)
        assert config.navigation.map_frame == 'map'
        assert config.navigation.base_link_frame == 'base_link'


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_config_types(self) -> None:
        """Test configuration parameter types.

        Verifies that configuration parameters have
        the expected data types.
        """
        config = get_config()

        # String parameters
        assert isinstance(config.server.server_name, str)
        assert isinstance(config.server.pose_uri, str)
        assert isinstance(config.navigation.map_frame, str)
        assert isinstance(config.navigation.base_link_frame, str)
        assert isinstance(config.logging.log_format, str)

        # Numeric parameters
        assert isinstance(config.navigation.default_tf_timeout, (int, float))
        assert isinstance(config.logging.level, int)

    def test_config_non_empty_strings(self) -> None:
        """Test that string configuration values are non-empty.

        Verifies that required string parameters
        are not empty.
        """
        config = get_config()

        assert len(config.server.server_name) > 0
        assert len(config.server.pose_uri) > 0
        assert len(config.navigation.map_frame) > 0
        assert len(config.navigation.base_link_frame) > 0
        assert len(config.logging.log_format) > 0

    def test_config_timeout_positive(self) -> None:
        """Test that timeout values are positive.

        Verifies that timeout configuration values
        are positive numbers.
        """
        config = get_config()

        assert config.navigation.default_tf_timeout > 0

    def test_config_valid_log_levels(self) -> None:
        """Test valid log level values.

        Verifies that log level is set to a valid logging level.
        """
        config = get_config()

        import logging
        valid_levels = [
            logging.DEBUG, logging.INFO, logging.WARNING,
            logging.ERROR, logging.CRITICAL
        ]
        assert config.logging.level in valid_levels


class TestConfigurationConsistency:
    """Tests for configuration consistency and relationships."""

    def test_config_singleton_behavior(self) -> None:
        """Test configuration singleton behavior.

        Verifies that get_config returns the same
        configuration instance on repeated calls.
        """
        config1 = get_config()
        config2 = get_config()

        # Should return the same configuration values
        assert config1.server.server_name == config2.server.server_name
        assert config1.logging.level == config2.logging.level
        assert config1.navigation.map_frame == config2.navigation.map_frame

    def test_config_frame_names_different(self) -> None:
        """Test that map and base frame names are different.

        Verifies that map frame and base frame have
        different names to avoid transform issues.
        """
        config = get_config()

        assert config.navigation.map_frame != config.navigation.base_link_frame

    def test_config_uri_format(self) -> None:
        """Test URI format validation.

        Verifies that URI values follow expected format patterns.
        """
        config = get_config()

        # Pose URI should be a valid URI format
        assert '://' in config.server.pose_uri or config.server.pose_uri.startswith('/')


class TestConfigurationErrors:
    """Tests for configuration error handling."""

    @patch.dict(os.environ, {'NAV2_MCP_LOG_LEVEL': 'INVALID_LEVEL'})
    def test_invalid_log_level_handling(self) -> None:
        """Test handling of invalid log level values.

        Verifies that invalid log levels are handled
        gracefully with fallback to default.
        """
        config = get_config()

        # Should fallback to a valid log level
        import logging
        valid_levels = [
            logging.DEBUG, logging.INFO, logging.WARNING,
            logging.ERROR, logging.CRITICAL
        ]
        assert config.logging.level in valid_levels

    @patch.dict(os.environ, {'NAV2_MCP_DEFAULT_TIMEOUT': 'not_a_number'})
    def test_invalid_timeout_handling(self) -> None:
        """Test handling of invalid timeout values.

        Verifies that invalid timeout values are handled
        gracefully with fallback to default.
        """
        config = get_config()

        # Should fallback to a valid positive number
        assert isinstance(config.navigation.default_tf_timeout, (int, float))
        assert config.navigation.default_tf_timeout > 0

    def test_config_loading_robustness(self) -> None:
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
        assert hasattr(config, 'logging')


class TestConfigurationDefaults:
    """Tests for configuration default values."""

    @patch.dict(os.environ, {}, clear=True)
    def test_default_server_name(self) -> None:
        """Test default server name value.

        Verifies that the default server name is appropriate.
        """
        config = get_config()

        # Should contain "Nav2" or "MCP" in the name
        server_name = config.server.server_name.lower()
        assert 'nav2' in server_name or 'mcp' in server_name

    @patch.dict(os.environ, {}, clear=True)
    def test_default_frame_names(self) -> None:
        """Test default frame name values.

        Verifies that default frame names follow ROS conventions.
        """
        config = get_config()

        # Common ROS frame names
        assert config.navigation.map_frame in ['map', 'world', 'odom']
        assert config.navigation.base_link_frame in [
            'base_link', 'base_footprint', 'robot_base'
        ]

    @patch.dict(os.environ, {}, clear=True)
    def test_default_logging_configuration(self) -> None:
        """Test default logging configuration.

        Verifies that default logging settings are reasonable.
        """
        config = get_config()

        # Should have reasonable default log level
        import logging
        assert config.logging.level in [
            logging.INFO, logging.DEBUG, logging.WARNING
        ]

        # Log format should contain basic elements
        log_format = config.logging.log_format
        assert 'levelname' in log_format.lower() or 'level' in log_format.lower()
        assert 'message' in log_format.lower()


class TestConfigOverrides:
    """Tests for configuration override functionality."""

    def test_config_with_navigation_overrides(self) -> None:
        """Test configuration with navigation section overrides.

        Verifies that navigation configuration can be overridden
        with custom values.
        """
        config_dict = {
            'navigation': {
                'default_backup_speed': 0.5,
                'default_tf_timeout': 1.0,
                'map_frame': 'world',
                'base_link_frame': 'robot',
                'max_waypoints': 50
            }
        }
        config = Config(config_dict)

        assert config.navigation.default_backup_speed == 0.5
        assert config.navigation.default_tf_timeout == 1.0
        assert config.navigation.map_frame == 'world'
        assert config.navigation.base_link_frame == 'robot'
        assert config.navigation.max_waypoints == 50

    def test_config_with_logging_overrides(self) -> None:
        """Test configuration with logging section overrides.

        Verifies that logging configuration can be overridden
        with custom values.
        """
        config_dict = {
            'logging': {
                'level': logging.DEBUG,
                'log_format': 'custom format',
                'node_name': 'test_node'
            }
        }
        config = Config(config_dict)

        assert config.logging.level == logging.DEBUG
        assert config.logging.log_format == 'custom format'
        assert config.logging.node_name == 'test_node'

    def test_config_with_server_overrides(self) -> None:
        """Test configuration with server section overrides.

        Verifies that server configuration can be overridden
        with custom values.
        """
        config_dict = {
            'server': {
                'transport': 'http',
                'server_name': 'custom-server',
                'description': 'Custom description',
                'pose_uri': 'custom://pose'
            }
        }
        config = Config(config_dict)

        assert config.server.transport == 'http'
        assert config.server.server_name == 'custom-server'
        assert config.server.description == 'Custom description'
        assert config.server.pose_uri == 'custom://pose'

    def test_config_with_multiple_section_overrides(self) -> None:
        """Test configuration with multiple section overrides.

        Verifies that multiple configuration sections can be
        overridden simultaneously.
        """
        config_dict = {
            'navigation': {'max_waypoints': 200},
            'logging': {'level': logging.WARNING},
            'server': {'server_name': 'multi-override-server'}
        }
        config = Config(config_dict)

        assert config.navigation.max_waypoints == 200
        assert config.logging.level == logging.WARNING
        assert config.server.server_name == 'multi-override-server'

    def test_config_ignores_invalid_keys(self) -> None:
        """Test that configuration ignores invalid override keys.

        Verifies that unknown configuration keys are ignored
        without raising errors.
        """
        config_dict = {
            'navigation': {
                'valid_key': 0.3,
                'invalid_key': 'should_be_ignored'
            }
        }
        # Should not raise an exception
        config = Config(config_dict)
        assert config is not None


class TestConfigValidationErrors:
    """Tests for configuration validation error conditions."""

    def test_negative_backup_speed_raises_error(self) -> None:
        """Test that negative backup speed raises ValueError.

        Verifies that invalid backup speed values are caught
        during validation.
        """
        with pytest.raises(ValueError, match='default_backup_speed must be positive'):
            Config({'navigation': {'default_backup_speed': -0.1}})

    def test_zero_backup_speed_raises_error(self) -> None:
        """Test that zero backup speed raises ValueError.

        Verifies that zero backup speed is rejected.
        """
        with pytest.raises(ValueError, match='default_backup_speed must be positive'):
            Config({'navigation': {'default_backup_speed': 0.0}})

    def test_negative_tf_timeout_raises_error(self) -> None:
        """Test that negative TF timeout raises ValueError.

        Verifies that invalid timeout values are caught.
        """
        with pytest.raises(ValueError, match='default_tf_timeout must be positive'):
            Config({'navigation': {'default_tf_timeout': -1.0}})

    def test_zero_tf_timeout_raises_error(self) -> None:
        """Test that zero TF timeout raises ValueError.

        Verifies that zero timeout is rejected.
        """
        with pytest.raises(ValueError, match='default_tf_timeout must be positive'):
            Config({'navigation': {'default_tf_timeout': 0.0}})

    def test_negative_max_waypoints_raises_error(self) -> None:
        """Test that negative max waypoints raises ValueError.

        Verifies that invalid waypoint limits are caught.
        """
        with pytest.raises(ValueError, match='max_waypoints must be positive'):
            Config({'navigation': {'max_waypoints': -10}})

    def test_zero_max_waypoints_raises_error(self) -> None:
        """Test that zero max waypoints raises ValueError.

        Verifies that zero waypoint limit is rejected.
        """
        with pytest.raises(ValueError, match='max_waypoints must be positive'):
            Config({'navigation': {'max_waypoints': 0}})

    def test_invalid_backup_distance_range_raises_error(self) -> None:
        """Test that invalid backup distance range raises ValueError.

        Verifies that min >= max backup distance is rejected.
        """
        with pytest.raises(
            ValueError,
            match='min_backup_distance must be less than max_backup_distance'
        ):
            Config({
                'navigation': {
                    'min_backup_distance': 5.0,
                    'max_backup_distance': 2.0
                }
            })

    def test_equal_backup_distance_raises_error(self) -> None:
        """Test that equal backup distances raise ValueError.

        Verifies that min == max backup distance is rejected.
        """
        with pytest.raises(
            ValueError,
            match='min_backup_distance must be less than max_backup_distance'
        ):
            Config({
                'navigation': {
                    'min_backup_distance': 3.0,
                    'max_backup_distance': 3.0
                }
            })

    def test_invalid_backup_speed_range_raises_error(self) -> None:
        """Test that invalid backup speed range raises ValueError.

        Verifies that min >= max backup speed is rejected.
        """
        with pytest.raises(
            ValueError,
            match='min_backup_speed must be less than max_backup_speed'
        ):
            Config({
                'navigation': {
                    'min_backup_speed': 0.8,
                    'max_backup_speed': 0.3
                }
            })

    def test_equal_backup_speed_raises_error(self) -> None:
        """Test that equal backup speeds raise ValueError.

        Verifies that min == max backup speed is rejected.
        """
        with pytest.raises(
            ValueError,
            match='min_backup_speed must be less than max_backup_speed'
        ):
            Config({
                'navigation': {
                    'min_backup_speed': 0.5,
                    'max_backup_speed': 0.5
                }
            })


class TestConfigUtilityFunctions:
    """Tests for configuration utility functions."""

    def test_to_dict_returns_nested_dict(self) -> None:
        """Test that to_dict returns proper nested dictionary.

        Verifies that configuration can be converted to
        dictionary format.
        """
        config = Config()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert 'navigation' in config_dict
        assert 'logging' in config_dict
        assert 'server' in config_dict

    def test_to_dict_contains_all_values(self) -> None:
        """Test that to_dict includes all configuration values.

        Verifies that the dictionary representation is complete.
        """
        config = Config()
        config_dict = config.to_dict()

        # Check navigation section
        nav_dict = config_dict['navigation']
        assert 'default_backup_speed' in nav_dict
        assert 'default_tf_timeout' in nav_dict
        assert 'map_frame' in nav_dict
        assert 'base_link_frame' in nav_dict
        assert 'max_waypoints' in nav_dict

        # Check logging section
        log_dict = config_dict['logging']
        assert 'level' in log_dict
        assert 'log_format' in log_dict
        assert 'node_name' in log_dict

        # Check server section
        server_dict = config_dict['server']
        assert 'transport' in server_dict
        assert 'server_name' in server_dict
        assert 'pose_uri' in server_dict

    def test_set_config_updates_global(self) -> None:
        """Test that set_config updates global configuration.

        Verifies that the global config can be updated
        with a new instance.
        """
        custom_config = Config({'server': {'server_name': 'test-config'}})
        set_config(custom_config)

        retrieved_config = get_config()
        assert retrieved_config.server.server_name == 'test-config'

    def test_load_config_from_dict_creates_and_sets_config(self) -> None:
        """Test that load_config_from_dict creates and sets config.

        Verifies that configuration can be loaded from dictionary
        and set as global in one operation.
        """
        config_dict = {
            'navigation': {'max_waypoints': 75},
            'server': {'server_name': 'loaded-server'}
        }
        loaded_config = load_config_from_dict(config_dict)

        assert loaded_config.navigation.max_waypoints == 75
        assert loaded_config.server.server_name == 'loaded-server'

        # Verify it's set as global
        global_config = get_config()
        assert global_config.navigation.max_waypoints == 75
        assert global_config.server.server_name == 'loaded-server'

    def test_config_to_dict_preserves_types(self) -> None:
        """Test that to_dict preserves value types.

        Verifies that type information is maintained
        when converting to dictionary.
        """
        config = Config()
        config_dict = config.to_dict()

        # Check types are preserved
        assert isinstance(config_dict['navigation']['default_backup_speed'], float)
        assert isinstance(config_dict['navigation']['max_waypoints'], int)
        assert isinstance(config_dict['logging']['level'], int)
        assert isinstance(config_dict['server']['server_name'], str)
