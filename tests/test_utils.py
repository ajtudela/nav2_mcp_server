"""Tests for utility functions and classes.

This module tests utility functions including MCPContextManager,
logging setup, JSON serialization, and helper functions.
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from nav2_mcp_server.utils import (
    MCPContextManager,
    safe_json_dumps,
    setup_logging,
    with_context_logging,
)


class TestMCPContextManager:
    """Tests for MCPContextManager class."""

    def test_mcp_context_manager_init(self) -> None:
        """Test MCPContextManager initialization.

        Verifies that MCPContextManager initializes properly
        with default values.
        """
        context_manager = MCPContextManager()

        assert context_manager is not None
        assert hasattr(context_manager, 'info')
        assert hasattr(context_manager, 'warning')
        assert hasattr(context_manager, 'error')

    def test_mcp_context_manager_info(self) -> None:
        """Test MCPContextManager info method.

        Verifies that info messages are handled correctly.
        """
        context_manager = MCPContextManager()

        # Should not raise exception
        context_manager.info_sync('Test info message')

    def test_mcp_context_manager_warning(self) -> None:
        """Test MCPContextManager warning method.

        Verifies that warning messages are handled correctly.
        """
        context_manager = MCPContextManager()

        # Should not raise exception
        context_manager.warning_sync('Test warning message')

    def test_mcp_context_manager_error(self) -> None:
        """Test MCPContextManager error method.

        Verifies that error messages are handled correctly.
        """
        context_manager = MCPContextManager()

        # Should not raise exception when using sync methods
        context_manager.error_sync('Test error message')

    @patch('anyio.from_thread.run')
    def test_mcp_context_manager_with_real_context(self, mock_from_thread_run: Mock) -> None:
        """Test MCPContextManager with real MCP Context.

        Verifies that MCPContextManager works with actual Context objects using sync methods.
        """
        # Mock a real MCP Context
        mock_context = Mock()
        mock_context.info = Mock()
        mock_context.warning = Mock()
        mock_context.error = Mock()

        context_manager = MCPContextManager(mock_context)

        # Use sync methods which call anyio.from_thread.run
        context_manager.info_sync('Test message')
        context_manager.warning_sync('Warning message')
        context_manager.error_sync('Error message')

        # The logger should have been called
        assert context_manager.logger is not None


class TestLoggingSetup:
    """Tests for logging setup functionality."""

    @patch('nav2_mcp_server.utils.logging.basicConfig')
    @patch('nav2_mcp_server.utils.get_config')
    def test_setup_logging_default(self, mock_get_config: Mock, mock_basic_config: Mock) -> None:
        """Test default logging setup.

        Verifies that logging is configured with default settings.
        """
        # Mock config
        mock_config = Mock()
        mock_config.logging.node_name = 'nav2_mcp_server'
        mock_config.logging.log_level = 'INFO'
        mock_config.logging.log_format = '%(asctime)s - %(levelname)s - %(message)s'
        mock_get_config.return_value = mock_config

        logger = setup_logging()

        assert logger is not None
        mock_basic_config.assert_called_once()

    @patch('nav2_mcp_server.utils.logging.basicConfig')
    @patch('nav2_mcp_server.utils.get_config')
    def test_setup_logging_debug_level(
        self, mock_get_config: Mock, mock_basic_config: Mock
    ) -> None:
        """Test logging setup with debug level.

        Verifies that debug logging level is correctly configured.
        """
        # Mock config with debug level
        mock_config = Mock()
        mock_config.logging.node_name = 'nav2_mcp_server'
        mock_config.logging.log_level = 'DEBUG'
        mock_config.logging.log_format = '%(asctime)s - %(levelname)s - %(message)s'
        mock_get_config.return_value = mock_config

        logger = setup_logging()

        assert logger is not None
        mock_basic_config.assert_called_once()

    @patch('nav2_mcp_server.utils.logging.getLogger')
    @patch('nav2_mcp_server.utils.get_config')
    def test_setup_logging_returns_logger(
        self, mock_get_config: Mock, mock_get_logger: Mock
    ) -> None:
        """Test that setup_logging returns a logger instance.

        Verifies that the function returns a proper logger object.
        """
        mock_config = Mock()
        mock_config.logging.node_name = 'nav2_mcp_server'
        mock_config.logging.log_level = 'INFO'
        mock_config.logging.log_format = '%(asctime)s - %(levelname)s - %(message)s'
        mock_get_config.return_value = mock_config

        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        result = setup_logging()

        assert result is mock_logger
        mock_get_logger.assert_called_once_with('nav2_mcp_server')


class TestSafeJSONDumps:
    """Tests for safe JSON serialization."""

    def test_safe_json_dumps_simple_dict(self) -> None:
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

    def test_safe_json_dumps_nested_dict(self) -> None:
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

    def test_safe_json_dumps_with_none_values(self) -> None:
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

    def test_safe_json_dumps_error_handling(self) -> None:
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

    def test_safe_json_dumps_with_formatting(self) -> None:
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

    def test_with_context_logging_decorator(self) -> None:
        """Test context logging decorator functionality.

        Verifies that the decorator properly wraps functions
        and adds context logging.
        """
        # Mock function to decorate
        @with_context_logging
        def test_function(x: int, y: int, context_manager: MCPContextManager | None = None) -> str:
            return f'success: {x} + {y}'

        # The decorator extracts ctx from kwargs and creates context_manager
        result = test_function(1, 2, context_manager=None)

        assert 'success' in result

    def test_with_context_logging_with_exception(self) -> None:
        """Test context logging decorator with exception.

        Verifies that exceptions are properly caught and returned as JSON.
        """
        from nav2_mcp_server.exceptions import NavigationError, NavigationErrorCode

        @with_context_logging
        def failing_function(context_manager: MCPContextManager | None = None) -> str:
            raise NavigationError(
                'Test error',
                NavigationErrorCode.NAVIGATION_FAILED,
                {}
            )

        # Should return JSON error instead of raising
        result = failing_function(context_manager=None)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert 'error' in parsed or 'message' in parsed

    def test_with_context_logging_generic_exception(self) -> None:
        """Test context logging decorator with generic exception.

        Verifies that generic exceptions are properly caught and returned as JSON.
        """
        @with_context_logging
        def failing_function(context_manager: MCPContextManager | None = None) -> str:
            raise ValueError('Test error')

        # Should return JSON error instead of raising
        result = failing_function(context_manager=None)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert 'error' in parsed or 'message' in parsed


class TestUtilityFunctions:
    """Tests for miscellaneous utility functions."""

    def test_utility_functions_exist(self) -> None:
        """Test that expected utility functions exist.

        Verifies that the module provides expected utility functions.
        """
        from nav2_mcp_server import utils

        # Check that expected functions/classes exist
        assert hasattr(utils, 'MCPContextManager')
        assert hasattr(utils, 'setup_logging')
        assert hasattr(utils, 'safe_json_dumps')
        assert hasattr(utils, 'with_context_logging')

    def test_import_structure(self) -> None:
        """Test import structure of utils module.

        Verifies that the utils module can be imported correctly.
        """
        # Should be able to import the module
        import nav2_mcp_server.utils

        # Module should exist
        assert nav2_mcp_server.utils is not None


class TestErrorHandling:
    """Tests for error handling in utility functions."""

    def test_mcp_context_manager_with_broken_context(self) -> None:
        """Test MCPContextManager with broken context object.

        Verifies that broken context objects cause expected exceptions.
        """
        # Create a mock context that raises exceptions
        broken_context = Mock()
        broken_context.info.side_effect = Exception('Context broken')

        context_manager = MCPContextManager(broken_context)

        # Should raise exception when context is broken
        with pytest.raises(Exception):
            context_manager.info_sync('Test message')

    def test_safe_json_dumps_circular_reference(self) -> None:
        """Test JSON serialization with circular reference.

        Verifies that circular references are handled gracefully.
        """
        # Create circular reference
        test_data: dict[str, object] = {}
        test_data['self'] = test_data

        result = safe_json_dumps(test_data)

        # Should return some JSON string, even if it's an error message
        assert isinstance(result, str)
        # Should be valid JSON
        parsed = json.loads(result)
        assert isinstance(parsed, dict)


class TestAsyncContextManagerMethods:
    """Tests for async methods of MCPContextManager."""

    async def test_async_info_with_context(self) -> None:
        """Test async info method with context.

        Verifies that async info method works correctly with context.
        """
        mock_context = Mock()
        mock_context.info = AsyncMock()

        context_manager = MCPContextManager(mock_context)

        await context_manager.info('Test async info message')

        # Context info should be called
        mock_context.info.assert_called_once_with('Test async info message')

    async def test_async_error_with_context(self) -> None:
        """Test async error method with context.

        Verifies that async error method works correctly with context.
        """
        mock_context = Mock()
        mock_context.error = AsyncMock()

        context_manager = MCPContextManager(mock_context)

        await context_manager.error('Test async error message')

        # Context error should be called
        mock_context.error.assert_called_once_with('Test async error message')

    async def test_async_warning_with_context(self) -> None:
        """Test async warning method with context.

        Verifies that async warning method works correctly with context.
        """
        mock_context = Mock()
        mock_context.warning = AsyncMock()

        context_manager = MCPContextManager(mock_context)

        await context_manager.warning('Test async warning message')

        # Context warning should be called
        mock_context.warning.assert_called_once_with('Test async warning message')

    async def test_async_info_without_context(self) -> None:
        """Test async info method without context.

        Verifies that async info method works without context.
        """
        context_manager = MCPContextManager()

        # Should not raise exception
        await context_manager.info('Test message without context')

    async def test_async_error_without_context(self) -> None:
        """Test async error method without context.

        Verifies that async error method works without context.
        """
        context_manager = MCPContextManager()

        # Should not raise exception
        await context_manager.error('Test error without context')

    async def test_async_warning_without_context(self) -> None:
        """Test async warning method without context.

        Verifies that async warning method works without context.
        """
        context_manager = MCPContextManager()

        # Should not raise exception
        await context_manager.warning('Test warning without context')


class TestNav2ActiveCheck:
    """Tests for Nav2 active check decorator."""

    @patch('nav2_mcp_server.navigation.get_navigator')
    def test_with_nav2_active_check_decorator(self, mock_get_nav: Mock) -> None:
        """Test Nav2 active check decorator functionality.

        Verifies that the decorator properly checks Nav2 activation.
        """
        from nav2_mcp_server.utils import with_nav2_active_check

        @with_nav2_active_check
        def test_function(x: int, y: int, context_manager: MCPContextManager | None = None) -> str:
            return f'Nav2 active: {x} + {y}'

        # Mock the navigator - patch where it's imported (inside the decorator)
        mock_navigator = Mock()
        mock_navigator.waitUntilNav2Active = Mock()
        mock_get_nav.return_value = mock_navigator

        result = test_function(1, 2, context_manager=None)

        assert 'Nav2 active' in result
        mock_navigator.waitUntilNav2Active.assert_called_once()

    @patch('anyio.from_thread.run')
    @patch('nav2_mcp_server.navigation.get_navigator')
    def test_with_nav2_active_check_with_context(
        self,
        mock_get_nav: Mock,
        mock_from_thread_run: Mock
    ) -> None:
        """Test Nav2 active check with context manager.

        Verifies that context logging works during Nav2 activation.
        """
        from nav2_mcp_server.utils import with_nav2_active_check

        @with_nav2_active_check
        def test_function(context_manager: MCPContextManager | None = None) -> str:
            return 'success'

        # Create a real context manager
        context_manager = MCPContextManager()

        mock_navigator = Mock()
        mock_navigator.waitUntilNav2Active = Mock()
        mock_get_nav.return_value = mock_navigator

        result = test_function(context_manager=context_manager)

        assert result == 'success'
        mock_navigator.waitUntilNav2Active.assert_called_once()

    @patch('nav2_mcp_server.navigation.get_navigator')
    def test_with_nav2_active_check_failure(self, mock_get_nav: Mock) -> None:
        """Test Nav2 active check decorator with activation failure.

        Verifies that exceptions during Nav2 activation are handled.
        """
        from nav2_mcp_server.exceptions import ROSError
        from nav2_mcp_server.utils import with_nav2_active_check

        @with_nav2_active_check
        def test_function(context_manager: MCPContextManager | None = None) -> str:
            return 'should not reach here'

        mock_navigator = Mock()
        mock_navigator.waitUntilNav2Active.side_effect = Exception('Nav2 not available')
        mock_get_nav.return_value = mock_navigator

        with pytest.raises(ROSError, match='Failed to activate Nav2'):
            test_function(context_manager=None)

    @patch('anyio.from_thread.run')
    @patch('nav2_mcp_server.navigation.get_navigator')
    def test_with_nav2_active_check_failure_with_context(
        self,
        mock_get_nav: Mock,
        mock_from_thread_run: Mock
    ) -> None:
        """Test Nav2 active check failure with context logging.

        Verifies that errors are logged to context when Nav2 activation fails.
        """
        from nav2_mcp_server.exceptions import ROSError
        from nav2_mcp_server.utils import with_nav2_active_check

        @with_nav2_active_check
        def test_function(context_manager: MCPContextManager | None = None) -> str:
            return 'should not reach here'

        context_manager = MCPContextManager()

        mock_navigator = Mock()
        mock_navigator.waitUntilNav2Active.side_effect = Exception('Nav2 timeout')
        mock_get_nav.return_value = mock_navigator

        with pytest.raises(ROSError, match='Failed to activate Nav2'):
            test_function(context_manager=context_manager)


class TestUtilityHelpers:
    """Tests for utility helper functions."""

    def test_create_success_message(self) -> None:
        """Test success message creation.

        Verifies that success messages are formatted correctly.
        """
        from nav2_mcp_server.utils import create_success_message

        result = create_success_message('navigate', {'x': 1.0, 'y': 2.0})

        assert 'Successfully navigate' in result
        assert 'x=1.0' in result
        assert 'y=2.0' in result

    def test_create_success_message_empty_details(self) -> None:
        """Test success message with empty details.

        Verifies that empty details are handled correctly.
        """
        from nav2_mcp_server.utils import create_success_message

        result = create_success_message('navigate', {})

        assert 'Successfully navigate' in result

    def test_create_failure_message(self) -> None:
        """Test failure message creation.

        Verifies that failure messages are formatted correctly.
        """
        from nav2_mcp_server.utils import create_failure_message

        result = create_failure_message(
            'navigate',
            'timeout',
            {'duration': 30}
        )

        assert 'Failed to navigate' in result
        assert 'timeout' in result
        assert 'duration=30' in result

    def test_create_failure_message_no_details(self) -> None:
        """Test failure message without details.

        Verifies that failure messages work without details.
        """
        from nav2_mcp_server.utils import create_failure_message

        result = create_failure_message('navigate', 'timeout')

        assert 'Failed to navigate' in result
        assert 'timeout' in result
        assert '(' not in result  # No details parentheses

    def test_validate_numeric_range_valid(self) -> None:
        """Test numeric range validation with valid value.

        Verifies that valid values pass validation.
        """
        from nav2_mcp_server.utils import validate_numeric_range

        # Should not raise exception
        validate_numeric_range(5.0, 0.0, 10.0, 'test_param')
        validate_numeric_range(0.0, 0.0, 10.0, 'test_param')
        validate_numeric_range(10.0, 0.0, 10.0, 'test_param')

    def test_validate_numeric_range_invalid(self) -> None:
        """Test numeric range validation with invalid value.

        Verifies that out-of-range values raise ValueError.
        """
        from nav2_mcp_server.utils import validate_numeric_range

        with pytest.raises(ValueError, match='test_param must be between'):
            validate_numeric_range(-1.0, 0.0, 10.0, 'test_param')

        with pytest.raises(ValueError, match='test_param must be between'):
            validate_numeric_range(11.0, 0.0, 10.0, 'test_param')
