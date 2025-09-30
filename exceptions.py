"""Custom exceptions for Nav2 MCP Server.

This module defines specific exception classes for different error conditions
that can occur during navigation operations.
"""

from typing import Optional, Any
from enum import Enum


class NavigationErrorCode(Enum):
    """Enumeration of navigation error codes."""

    UNKNOWN = 0
    NAVIGATION_FAILED = 1
    NAVIGATION_CANCELED = 2
    TRANSFORM_UNAVAILABLE = 3
    NAV2_NOT_ACTIVE = 4
    INVALID_WAYPOINTS = 5
    INVALID_PARAMETERS = 6
    TIMEOUT = 7
    ROS_ERROR = 8


class Nav2MCPError(Exception):
    """Base exception class for Nav2 MCP Server errors.

    Attributes
    ----------
    message : str
        Human-readable error message.
    error_code : NavigationErrorCode
        Specific error code for the exception.
    details : dict, optional
        Additional error details or context.
    """

    def __init__(
        self,
        message: str,
        error_code: NavigationErrorCode = NavigationErrorCode.UNKNOWN,
        details: Optional[dict] = None
    ):
        """Initialize the exception.

        Parameters
        ----------
        message : str
            Human-readable error message.
        error_code : NavigationErrorCode, optional
            Specific error code (default: UNKNOWN).
        details : dict, optional
            Additional error details or context.
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def to_dict(self) -> dict:
        """Convert exception to dictionary representation.

        Returns
        -------
        dict
            Dictionary containing error information.
        """
        return {
            'error': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code.name,
            'details': self.details
        }


class NavigationError(Nav2MCPError):
    """Exception raised when navigation operations fail."""

    def __init__(
        self,
        message: str,
        error_code: NavigationErrorCode = (
            NavigationErrorCode.NAVIGATION_FAILED
        ),
        details: Optional[dict] = None
    ):
        super().__init__(message, error_code, details)


class TransformError(Nav2MCPError):
    """Exception raised when transform operations fail."""

    def __init__(
        self,
        message: str,
        error_code: NavigationErrorCode = (
            NavigationErrorCode.TRANSFORM_UNAVAILABLE
        ),
        details: Optional[dict] = None
    ):
        super().__init__(message, error_code, details)


class ConfigurationError(Nav2MCPError):
    """Exception raised when configuration is invalid."""

    def __init__(
        self,
        message: str,
        error_code: NavigationErrorCode = (
            NavigationErrorCode.INVALID_PARAMETERS
        ),
        details: Optional[dict] = None
    ):
        super().__init__(message, error_code, details)


class TimeoutError(Nav2MCPError):
    """Exception raised when operations timeout."""

    def __init__(
        self,
        message: str,
        timeout_duration: Optional[float] = None,
        details: Optional[dict] = None
    ):
        error_details = details or {}
        if timeout_duration is not None:
            error_details['timeout_duration'] = timeout_duration

        super().__init__(message, NavigationErrorCode.TIMEOUT, error_details)


class ROSError(Nav2MCPError):
    """Exception raised when ROS operations fail."""

    def __init__(
        self,
        message: str,
        ros_exception: Optional[Exception] = None,
        details: Optional[dict] = None
    ):
        error_details = details or {}
        if ros_exception is not None:
            error_details['ros_exception_type'] = type(ros_exception).__name__
            error_details['ros_exception_message'] = str(ros_exception)

        super().__init__(message, NavigationErrorCode.ROS_ERROR, error_details)


def create_navigation_error_from_result(
    result: Any, operation: str
) -> NavigationError:
    """Create a NavigationError based on navigation result.

    Parameters
    ----------
    result : Any
        The navigation result from Nav2.
    operation : str
        Description of the operation that failed.

    Returns
    -------
    NavigationError
        Appropriate NavigationError for the result.
    """
    from nav2_simple_commander.robot_navigator import TaskResult

    if result == TaskResult.FAILED:
        return NavigationError(
            f"{operation} failed",
            NavigationErrorCode.NAVIGATION_FAILED,
            {'task_result': 'FAILED'}
        )
    elif result == TaskResult.CANCELED:
        return NavigationError(
            f"{operation} was canceled",
            NavigationErrorCode.NAVIGATION_CANCELED,
            {'task_result': 'CANCELED'}
        )
    else:
        return NavigationError(
            f"{operation} completed with unexpected result: {result}",
            NavigationErrorCode.UNKNOWN,
            {'task_result': str(result)}
        )
