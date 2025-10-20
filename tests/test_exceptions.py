"""Tests for exceptions module.

This module tests custom exception classes, error codes,
and error handling utilities.
"""

from unittest.mock import Mock

import pytest

from nav2_mcp_server.exceptions import (
    ConfigurationError,
    create_navigation_error_from_result,
    Nav2MCPError,
    NavigationError,
    NavigationErrorCode,
    ROSError,
    TimeoutError,
    TransformError,
)


class TestNavigationErrorCode:
    """Tests for NavigationErrorCode enumeration."""

    def test_error_code_values_exist(self) -> None:
        """Test that all error codes are defined.

        Verifies that the enum contains all expected error codes.
        """
        assert hasattr(NavigationErrorCode, 'UNKNOWN')
        assert hasattr(NavigationErrorCode, 'NAVIGATION_FAILED')
        assert hasattr(NavigationErrorCode, 'NAVIGATION_CANCELED')
        assert hasattr(NavigationErrorCode, 'TRANSFORM_UNAVAILABLE')
        assert hasattr(NavigationErrorCode, 'NAV2_NOT_ACTIVE')
        assert hasattr(NavigationErrorCode, 'INVALID_WAYPOINTS')
        assert hasattr(NavigationErrorCode, 'INVALID_PARAMETERS')
        assert hasattr(NavigationErrorCode, 'TIMEOUT')
        assert hasattr(NavigationErrorCode, 'ROS_ERROR')

    def test_error_code_values_are_unique(self) -> None:
        """Test that error code values are unique.

        Verifies that each error code has a distinct value.
        """
        codes = [code.value for code in NavigationErrorCode]
        assert len(codes) == len(set(codes))

    def test_error_code_names(self) -> None:
        """Test error code name access.

        Verifies that error code names can be accessed.
        """
        assert NavigationErrorCode.UNKNOWN.name == 'UNKNOWN'
        assert NavigationErrorCode.NAVIGATION_FAILED.name == 'NAVIGATION_FAILED'
        assert NavigationErrorCode.TIMEOUT.name == 'TIMEOUT'


class TestNav2MCPError:
    """Tests for base Nav2MCPError exception class."""

    def test_basic_exception_creation(self) -> None:
        """Test basic exception creation.

        Verifies that base exception can be created with
        minimal parameters.
        """
        error = Nav2MCPError('Test error')

        assert str(error) == 'Test error'
        assert error.message == 'Test error'
        assert error.error_code == NavigationErrorCode.UNKNOWN
        assert error.details == {}

    def test_exception_with_error_code(self) -> None:
        """Test exception creation with error code.

        Verifies that specific error codes can be set.
        """
        error = Nav2MCPError(
            'Navigation failed',
            error_code=NavigationErrorCode.NAVIGATION_FAILED
        )

        assert error.message == 'Navigation failed'
        assert error.error_code == NavigationErrorCode.NAVIGATION_FAILED

    def test_exception_with_details(self) -> None:
        """Test exception creation with details.

        Verifies that additional details can be attached.
        """
        details = {'waypoint': 5, 'reason': 'obstacle'}
        error = Nav2MCPError('Error with details', details=details)

        assert error.details == details
        assert error.details['waypoint'] == 5
        assert error.details['reason'] == 'obstacle'

    def test_exception_with_all_parameters(self) -> None:
        """Test exception with all parameters.

        Verifies that all parameters can be set together.
        """
        details = {'param': 'value'}
        error = Nav2MCPError(
            'Complete error',
            error_code=NavigationErrorCode.INVALID_PARAMETERS,
            details=details
        )

        assert error.message == 'Complete error'
        assert error.error_code == NavigationErrorCode.INVALID_PARAMETERS
        assert error.details == details

    def test_to_dict_basic(self) -> None:
        """Test to_dict method with basic exception.

        Verifies that exception can be converted to dictionary.
        """
        error = Nav2MCPError('Test error')
        error_dict = error.to_dict()

        assert isinstance(error_dict, dict)
        assert error_dict['error'] == 'Nav2MCPError'
        assert error_dict['message'] == 'Test error'
        assert error_dict['error_code'] == 'UNKNOWN'
        assert error_dict['details'] == {}

    def test_to_dict_with_details(self) -> None:
        """Test to_dict method with details.

        Verifies that details are included in dictionary.
        """
        details = {'key': 'value', 'number': 42}
        error = Nav2MCPError(
            'Error with details',
            error_code=NavigationErrorCode.TIMEOUT,
            details=details
        )
        error_dict = error.to_dict()

        assert error_dict['message'] == 'Error with details'
        assert error_dict['error_code'] == 'TIMEOUT'
        assert error_dict['details'] == details

    def test_exception_is_raisable(self) -> None:
        """Test that exception can be raised and caught.

        Verifies that the exception works in try/except blocks.
        """
        with pytest.raises(Nav2MCPError) as exc_info:
            raise Nav2MCPError('Raised error')

        assert str(exc_info.value) == 'Raised error'


class TestNavigationError:
    """Tests for NavigationError exception class."""

    def test_navigation_error_default_code(self) -> None:
        """Test NavigationError default error code.

        Verifies that NavigationError has appropriate default code.
        """
        error = NavigationError('Navigation failed')

        assert error.error_code == NavigationErrorCode.NAVIGATION_FAILED

    def test_navigation_error_custom_code(self) -> None:
        """Test NavigationError with custom error code.

        Verifies that error code can be customized.
        """
        error = NavigationError(
            'Navigation canceled',
            error_code=NavigationErrorCode.NAVIGATION_CANCELED
        )

        assert error.error_code == NavigationErrorCode.NAVIGATION_CANCELED

    def test_navigation_error_with_details(self) -> None:
        """Test NavigationError with details.

        Verifies that details can be attached to navigation errors.
        """
        details = {'position': [1.0, 2.0], 'orientation': 0.5}
        error = NavigationError('Failed to reach goal', details=details)

        assert error.details == details

    def test_navigation_error_inheritance(self) -> None:
        """Test NavigationError inheritance.

        Verifies that NavigationError is a Nav2MCPError.
        """
        error = NavigationError('Test')

        assert isinstance(error, Nav2MCPError)
        assert isinstance(error, Exception)

    def test_navigation_error_to_dict(self) -> None:
        """Test NavigationError to_dict method.

        Verifies dictionary representation includes correct class name.
        """
        error = NavigationError('Nav error')
        error_dict = error.to_dict()

        assert error_dict['error'] == 'NavigationError'
        assert error_dict['error_code'] == 'NAVIGATION_FAILED'


class TestTransformError:
    """Tests for TransformError exception class."""

    def test_transform_error_default_code(self) -> None:
        """Test TransformError default error code.

        Verifies that TransformError has appropriate default code.
        """
        error = TransformError('Transform unavailable')

        assert error.error_code == NavigationErrorCode.TRANSFORM_UNAVAILABLE

    def test_transform_error_custom_code(self) -> None:
        """Test TransformError with custom error code.

        Verifies that error code can be customized.
        """
        error = TransformError(
            'Transform timeout',
            error_code=NavigationErrorCode.TIMEOUT
        )

        assert error.error_code == NavigationErrorCode.TIMEOUT

    def test_transform_error_with_details(self) -> None:
        """Test TransformError with frame details.

        Verifies that transform-specific details can be attached.
        """
        details = {'source_frame': 'map', 'target_frame': 'base_link'}
        error = TransformError('Cannot transform', details=details)

        assert error.details['source_frame'] == 'map'
        assert error.details['target_frame'] == 'base_link'

    def test_transform_error_inheritance(self) -> None:
        """Test TransformError inheritance.

        Verifies that TransformError is a Nav2MCPError.
        """
        error = TransformError('Test')

        assert isinstance(error, Nav2MCPError)

    def test_transform_error_to_dict(self) -> None:
        """Test TransformError to_dict method.

        Verifies dictionary representation.
        """
        error = TransformError('Transform error')
        error_dict = error.to_dict()

        assert error_dict['error'] == 'TransformError'
        assert error_dict['error_code'] == 'TRANSFORM_UNAVAILABLE'


class TestConfigurationError:
    """Tests for ConfigurationError exception class."""

    def test_configuration_error_default_code(self) -> None:
        """Test ConfigurationError default error code.

        Verifies that ConfigurationError has appropriate default code.
        """
        error = ConfigurationError('Invalid configuration')

        assert error.error_code == NavigationErrorCode.INVALID_PARAMETERS

    def test_configuration_error_with_details(self) -> None:
        """Test ConfigurationError with parameter details.

        Verifies that configuration-specific details can be attached.
        """
        details = {'parameter': 'max_speed', 'value': -1, 'constraint': 'positive'}
        error = ConfigurationError('Invalid parameter', details=details)

        assert error.details == details

    def test_configuration_error_inheritance(self) -> None:
        """Test ConfigurationError inheritance.

        Verifies that ConfigurationError is a Nav2MCPError.
        """
        error = ConfigurationError('Test')

        assert isinstance(error, Nav2MCPError)

    def test_configuration_error_to_dict(self) -> None:
        """Test ConfigurationError to_dict method.

        Verifies dictionary representation.
        """
        error = ConfigurationError('Config error')
        error_dict = error.to_dict()

        assert error_dict['error'] == 'ConfigurationError'
        assert error_dict['error_code'] == 'INVALID_PARAMETERS'


class TestTimeoutError:
    """Tests for TimeoutError exception class."""

    def test_timeout_error_basic(self) -> None:
        """Test basic TimeoutError creation.

        Verifies that TimeoutError can be created without duration.
        """
        error = TimeoutError('Operation timed out')

        assert error.message == 'Operation timed out'
        assert error.error_code == NavigationErrorCode.TIMEOUT
        assert error.details == {}

    def test_timeout_error_with_duration(self) -> None:
        """Test TimeoutError with timeout duration.

        Verifies that timeout duration is stored in details.
        """
        error = TimeoutError('Timed out', timeout_duration=5.0)

        assert error.details['timeout_duration'] == 5.0

    def test_timeout_error_with_details_and_duration(self) -> None:
        """Test TimeoutError with both details and duration.

        Verifies that duration is added to existing details.
        """
        details = {'operation': 'navigation'}
        error = TimeoutError(
            'Navigation timeout',
            timeout_duration=10.0,
            details=details
        )

        assert error.details['operation'] == 'navigation'
        assert error.details['timeout_duration'] == 10.0

    def test_timeout_error_without_duration_parameter(self) -> None:
        """Test TimeoutError when duration is None.

        Verifies that None duration is not added to details.
        """
        error = TimeoutError('Timeout', timeout_duration=None)

        assert 'timeout_duration' not in error.details

    def test_timeout_error_inheritance(self) -> None:
        """Test TimeoutError inheritance.

        Verifies that TimeoutError is a Nav2MCPError.
        """
        error = TimeoutError('Test')

        assert isinstance(error, Nav2MCPError)

    def test_timeout_error_to_dict(self) -> None:
        """Test TimeoutError to_dict method.

        Verifies dictionary representation includes timeout info.
        """
        error = TimeoutError('Timeout', timeout_duration=3.5)
        error_dict = error.to_dict()

        assert error_dict['error'] == 'TimeoutError'
        assert error_dict['error_code'] == 'TIMEOUT'
        assert error_dict['details']['timeout_duration'] == 3.5


class TestROSError:
    """Tests for ROSError exception class."""

    def test_ros_error_basic(self) -> None:
        """Test basic ROSError creation.

        Verifies that ROSError can be created without ROS exception.
        """
        error = ROSError('ROS operation failed')

        assert error.message == 'ROS operation failed'
        assert error.error_code == NavigationErrorCode.ROS_ERROR
        assert error.details == {}

    def test_ros_error_with_ros_exception(self) -> None:
        """Test ROSError with ROS exception details.

        Verifies that ROS exception info is stored in details.
        """
        ros_exception = ValueError('ROS value error')
        error = ROSError('ROS failed', ros_exception=ros_exception)

        assert error.details['ros_exception_type'] == 'ValueError'
        assert error.details['ros_exception_message'] == 'ROS value error'

    def test_ros_error_with_details_and_exception(self) -> None:
        """Test ROSError with both details and ROS exception.

        Verifies that exception info is added to existing details.
        """
        ros_exception = RuntimeError('Runtime issue')
        details = {'node': 'nav2_controller'}
        error = ROSError(
            'Controller error',
            ros_exception=ros_exception,
            details=details
        )

        assert error.details['node'] == 'nav2_controller'
        assert error.details['ros_exception_type'] == 'RuntimeError'
        assert error.details['ros_exception_message'] == 'Runtime issue'

    def test_ros_error_without_ros_exception(self) -> None:
        """Test ROSError when ros_exception is None.

        Verifies that None exception is not added to details.
        """
        error = ROSError('ROS error', ros_exception=None)

        assert 'ros_exception_type' not in error.details
        assert 'ros_exception_message' not in error.details

    def test_ros_error_inheritance(self) -> None:
        """Test ROSError inheritance.

        Verifies that ROSError is a Nav2MCPError.
        """
        error = ROSError('Test')

        assert isinstance(error, Nav2MCPError)

    def test_ros_error_to_dict(self) -> None:
        """Test ROSError to_dict method.

        Verifies dictionary representation includes ROS info.
        """
        ros_exception = TypeError('Type mismatch')
        error = ROSError('ROS error', ros_exception=ros_exception)
        error_dict = error.to_dict()

        assert error_dict['error'] == 'ROSError'
        assert error_dict['error_code'] == 'ROS_ERROR'
        assert error_dict['details']['ros_exception_type'] == 'TypeError'


class TestCreateNavigationErrorFromResult:
    """Tests for create_navigation_error_from_result utility function."""

    def test_create_error_from_failed_result(self) -> None:
        """Test error creation from FAILED task result.

        Verifies that FAILED result creates appropriate error.
        """
        # Mock TaskResult
        mock_task_result = Mock()
        mock_task_result.FAILED = 3
        mock_task_result.CANCELED = 2
        mock_task_result.SUCCEEDED = 1

        # Create a mock module
        import sys
        mock_module = Mock()
        mock_module.TaskResult = mock_task_result
        sys.modules['nav2_simple_commander'] = Mock()
        sys.modules['nav2_simple_commander.robot_navigator'] = mock_module

        try:
            error = create_navigation_error_from_result(
                mock_task_result.FAILED,
                'Test operation'
            )

            assert isinstance(error, NavigationError)
            assert error.error_code == NavigationErrorCode.NAVIGATION_FAILED
            assert 'Test operation failed' in error.message
            assert error.details['task_result'] == 'FAILED'
        finally:
            # Clean up mock
            if 'nav2_simple_commander' in sys.modules:
                del sys.modules['nav2_simple_commander']
            if 'nav2_simple_commander.robot_navigator' in sys.modules:
                del sys.modules['nav2_simple_commander.robot_navigator']

    def test_create_error_from_canceled_result(self) -> None:
        """Test error creation from CANCELED task result.

        Verifies that CANCELED result creates appropriate error.
        """
        # Mock TaskResult
        mock_task_result = Mock()
        mock_task_result.FAILED = 3
        mock_task_result.CANCELED = 2
        mock_task_result.SUCCEEDED = 1

        import sys
        mock_module = Mock()
        mock_module.TaskResult = mock_task_result
        sys.modules['nav2_simple_commander'] = Mock()
        sys.modules['nav2_simple_commander.robot_navigator'] = mock_module

        try:
            error = create_navigation_error_from_result(
                mock_task_result.CANCELED,
                'Test operation'
            )

            assert isinstance(error, NavigationError)
            assert error.error_code == NavigationErrorCode.NAVIGATION_CANCELED
            assert 'Test operation was canceled' in error.message
            assert error.details['task_result'] == 'CANCELED'
        finally:
            # Clean up mock
            if 'nav2_simple_commander' in sys.modules:
                del sys.modules['nav2_simple_commander']
            if 'nav2_simple_commander.robot_navigator' in sys.modules:
                del sys.modules['nav2_simple_commander.robot_navigator']

    def test_create_error_from_unknown_result(self) -> None:
        """Test error creation from unknown task result.

        Verifies that unknown results create error with UNKNOWN code.
        """
        # Mock TaskResult
        mock_task_result = Mock()
        mock_task_result.FAILED = 3
        mock_task_result.CANCELED = 2
        mock_task_result.SUCCEEDED = 1

        import sys
        mock_module = Mock()
        mock_module.TaskResult = mock_task_result
        sys.modules['nav2_simple_commander'] = Mock()
        sys.modules['nav2_simple_commander.robot_navigator'] = mock_module

        try:
            # Use a result value that's not FAILED or CANCELED
            unknown_result = 999
            error = create_navigation_error_from_result(
                unknown_result,
                'Test operation'
            )

            assert isinstance(error, NavigationError)
            assert error.error_code == NavigationErrorCode.UNKNOWN
            assert 'unexpected result' in error.message.lower()
            assert '999' in error.details['task_result']
        finally:
            # Clean up mock
            if 'nav2_simple_commander' in sys.modules:
                del sys.modules['nav2_simple_commander']
            if 'nav2_simple_commander.robot_navigator' in sys.modules:
                del sys.modules['nav2_simple_commander.robot_navigator']


class TestExceptionInteroperability:
    """Tests for exception interoperability and edge cases."""

    def test_exception_can_be_caught_as_base_exception(self) -> None:
        """Test that specific exceptions can be caught as base type.

        Verifies exception hierarchy works correctly.
        """
        with pytest.raises(Nav2MCPError):
            raise NavigationError('Test')

        with pytest.raises(Nav2MCPError):
            raise TransformError('Test')

        with pytest.raises(Nav2MCPError):
            raise ConfigurationError('Test')

    def test_exception_message_persistence(self) -> None:
        """Test that exception messages persist through raise/catch.

        Verifies message is preserved in exception handling.
        """
        original_message = 'Persistent error message'

        with pytest.raises(Nav2MCPError) as exc_info:
            raise Nav2MCPError(original_message)

        assert exc_info.value.message == original_message
        assert str(exc_info.value) == original_message

    def test_exception_details_mutability(self) -> None:
        """Test exception details dictionary mutability.

        Verifies that details can be modified after creation.
        """
        error = Nav2MCPError('Test', details={'key': 'value'})

        # Details should be mutable
        error.details['new_key'] = 'new_value'

        assert error.details['key'] == 'value'
        assert error.details['new_key'] == 'new_value'

    def test_multiple_exceptions_independent(self) -> None:
        """Test that multiple exception instances are independent.

        Verifies that exception instances don't share state.
        """
        error1 = Nav2MCPError('Error 1', details={'id': 1})
        error2 = Nav2MCPError('Error 2', details={'id': 2})

        assert error1.details['id'] == 1
        assert error2.details['id'] == 2
        assert error1.message != error2.message
