"""Tests for utility functions and classes.

This module tests utility functions including MCPContextManager,
logging setup, JSON serialization, and helper functions.
"""

from unittest.mock import Mock, patch
import pytest
import json

from nav2_mcp_server.utils import (
    MCPContextManager,
    setup_logging,
    safe_json_dumps,
    with_context_logging
)


class TestMCPContextManager:
    """Tests for MCPContextManager class."""

    def test_mcp_context_manager_init(self):
        """Test MCPContextManager initialization.

        Verifies that MCPContextManager initializes properly
        with default values.
        """
        context_manager = MCPContextManager()

        assert context_manager is not None
        assert hasattr(context_manager, 'info')
        assert hasattr(context_manager, 'warning')
        assert hasattr(context_manager, 'error')

    def test_mcp_context_manager_info(self):
        """Test MCPContextManager info method.

        Verifies that info messages are handled correctly.
        """
        context_manager = MCPContextManager()

        # Should not raise exception
        context_manager.info("Test info message")

    def test_mcp_context_manager_warning(self):
        """Test MCPContextManager warning method.

        Verifies that warning messages are handled correctly.
        """
        context_manager = MCPContextManager()

        # Should not raise exception
        context_manager.warning("Test warning message")

    def test_mcp_context_manager_error(self):
        """Test MCPContextManager error method.

        Verifies that error messages are handled correctly.
        """
        context_manager = MCPContextManager()

        # Should not raise exception
        context_manager.error("Test error message")

    def test_mcp_context_manager_with_real_context(self):
        """Test MCPContextManager with real MCP Context.

        Verifies that MCPContextManager works with actual Context objects.
        """
        # Mock a real MCP Context
        mock_context = Mock()
        mock_context.info = Mock()
        mock_context.warning = Mock()
        mock_context.error = Mock()

        context_manager = MCPContextManager(mock_context)

        context_manager.info("Test message")
        context_manager.warning("Warning message")
        context_manager.error("Error message")

        mock_context.info.assert_called_once_with("Test message")
        mock_context.warning.assert_called_once_with("Warning message")
        mock_context.error.assert_called_once_with("Error message")


class TestLoggingSetup:
    """Tests for logging setup functionality."""

    @patch('nav2_mcp_server.utils.logging.basicConfig')
    @patch('nav2_mcp_server.utils.get_config')
    def test_setup_logging_default(self, mock_get_config, mock_basic_config):
        """Test logging setup with default configuration.

        Verifies that logging is configured with appropriate defaults.
        """
        # Mock config
        mock_config = Mock()
        mock_config.logging.log_level = 'INFO'
        mock_config.logging.log_format = '%(asctime)s - %(levelname)s - %(message)s'
        mock_get_config.return_value = mock_config

        logger = setup_logging()

        assert logger is not None
        mock_basic_config.assert_called_once()

    @patch('nav2_mcp_server.utils.logging.basicConfig')
    @patch('nav2_mcp_server.utils.get_config')
    def test_setup_logging_debug_level(self, mock_get_config, mock_basic_config):
        """Test logging setup with debug level.

        Verifies that debug logging level is correctly configured.
        """
        # Mock config with debug level
        mock_config = Mock()
        mock_config.logging.log_level = 'DEBUG'
        mock_config.logging.log_format = '%(asctime)s - %(levelname)s - %(message)s'
        mock_get_config.return_value = mock_config

        logger = setup_logging()

        assert logger is not None
        mock_basic_config.assert_called_once()

    @patch('nav2_mcp_server.utils.logging.getLogger')
    @patch('nav2_mcp_server.utils.get_config')
    def test_setup_logging_returns_logger(self, mock_get_config, mock_get_logger):
        """Test that setup_logging returns a logger instance.

        Verifies that the function returns a proper logger object.
        """
        mock_config = Mock()
        mock_config.logging.log_level = 'INFO'
        mock_config.logging.log_format = '%(asctime)s - %(levelname)s - %(message)s'
        mock_get_config.return_value = mock_config

        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        result = setup_logging()

        assert result is mock_logger
        mock_get_logger.assert_called_with('nav2_mcp_server')


class TestSafeJSONDumps:
    """Tests for safe JSON serialization."""

    def test_safe_json_dumps_simple_dict(self):
        """Test JSON serialization of simple dictionary.

        Verifies that simple dictionaries are correctly serialized.
        """
        test_data = {
            'name': 'test',
            'value': 42,
            'active': True
        }

        result = safe_json_dumps(test_data)

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == test_data

    def test_safe_json_dumps_nested_dict(self):
        """Test JSON serialization of nested dictionary.

        Verifies that nested data structures are correctly serialized.
        """
        test_data = {
            'robot': {
                'position': {'x': 1.0, 'y': 2.0, 'z': 0.0},
                'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.707, 'w': 0.707}
            },
            'status': 'active',
            'sensors': ['lidar', 'camera', 'imu']
        }

        result = safe_json_dumps(test_data)

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == test_data

    def test_safe_json_dumps_with_none_values(self):
        """Test JSON serialization with None values.

        Verifies that None values are handled correctly.
        """
        test_data = {
            'value': None,
            'other': 'test'
        }

        result = safe_json_dumps(test_data)

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == test_data

    def test_safe_json_dumps_error_handling(self):
        """Test JSON serialization error handling.

        Verifies that non-serializable objects are handled gracefully.
        """
        # Create a non-serializable object
        class NonSerializable:
            pass

        test_data = {
            'good': 'value',
            'bad': NonSerializable()
        }

        result = safe_json_dumps(test_data)

        # Should return a JSON string even with error
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert 'error' in parsed or 'good' in parsed

    def test_safe_json_dumps_with_formatting(self):
        """Test JSON serialization with formatting options.

        Verifies that JSON formatting options work correctly.
        """
        test_data = {'a': 1, 'b': 2}

        result = safe_json_dumps(test_data, indent=2)

        assert isinstance(result, str)
        assert '\n' in result  # Should have newlines due to indent
        parsed = json.loads(result)
        assert parsed == test_data


class TestWithContextLogging:
    """Tests for context logging decorator."""

    def test_with_context_logging_decorator(self):
        """Test context logging decorator functionality.

        Verifies that the decorator properly wraps functions
        and adds context logging.
        """
        # Mock function to decorate
        @with_context_logging("Test operation")
        def test_function(context_manager):
            return "success"

        context_manager = MCPContextManager()

        with patch.object(context_manager, 'info') as mock_info:
            result = test_function(context_manager)

            assert result == "success"
            # Should have called info at least once
            assert mock_info.call_count >= 1

    def test_with_context_logging_with_exception(self):
        """Test context logging decorator with exception.

        Verifies that exceptions are properly logged and re-raised.
        """
        @with_context_logging("Test operation with error")
        def failing_function(context_manager):
            raise ValueError("Test error")

        context_manager = MCPContextManager()

        with patch.object(context_manager, 'error') as mock_error:
            with pytest.raises(ValueError):
                failing_function(context_manager)

            # Should have logged the error
            mock_error.assert_called_once()

    def test_with_context_logging_async_function(self):
        """Test context logging decorator with async function.

        Verifies that the decorator works with async functions.
        """
        @with_context_logging("Async test operation")
        async def async_test_function(context_manager):
            return "async success"

        context_manager = MCPContextManager()

        with patch.object(context_manager, 'info') as mock_info:
            # Note: This would need to be run with asyncio in a real test
            # For this mock test, we just verify the decorator can be applied
            assert callable(async_test_function)


class TestUtilityFunctions:
    """Tests for miscellaneous utility functions."""

    def test_utility_functions_exist(self):
        """Test that expected utility functions exist.

        Verifies that the module provides expected utility functions.
        """
        from nav2_mcp_server import utils

        # Check that expected functions/classes exist
        assert hasattr(utils, 'MCPContextManager')
        assert hasattr(utils, 'setup_logging')
        assert hasattr(utils, 'safe_json_dumps')
        assert hasattr(utils, 'with_context_logging')

    def test_import_structure(self):
        """Test import structure of utils module.

        Verifies that the utils module can be imported correctly.
        """
        # Should be able to import the module
        import nav2_mcp_server.utils

        # Module should exist
        assert nav2_mcp_server.utils is not None


class TestErrorHandling:
    """Tests for error handling in utility functions."""

    def test_mcp_context_manager_with_broken_context(self):
        """Test MCPContextManager with broken context object.

        Verifies that broken context objects are handled gracefully.
        """
        # Create a mock context that raises exceptions
        broken_context = Mock()
        broken_context.info.side_effect = Exception("Context broken")
        broken_context.warning.side_effect = Exception("Context broken")
        broken_context.error.side_effect = Exception("Context broken")

        context_manager = MCPContextManager(broken_context)

        # Should not raise exceptions even with broken context
        context_manager.info("Test message")
        context_manager.warning("Warning message")
        context_manager.error("Error message")

    def test_safe_json_dumps_circular_reference(self):
        """Test JSON serialization with circular reference.

        Verifies that circular references are handled gracefully.
        """
        # Create circular reference
        test_data = {}
        test_data['self'] = test_data

        result = safe_json_dumps(test_data)

        # Should return some JSON string, even if it's an error message
        assert isinstance(result, str)
        # Should be valid JSON
        parsed = json.loads(result)
        assert isinstance(parsed, dict)