# Copyright (c) 2025 Alberto J. Tudela Roldán
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for MCP context management and logging.

This module provides helpers for managing MCP context, logging,
and common async/sync operations.
"""

import functools
import json
import logging
from typing import Callable, Optional, TypeVar, Any, Dict
import anyio
from fastmcp import Context

from .config import get_config
from .exceptions import Nav2MCPError, ROSError


F = TypeVar('F', bound=Callable[..., Any])


class MCPContextManager:
    """Manager for MCP context operations and logging."""

    def __init__(self, ctx: Optional[Context] = None):
        """Initialize context manager.

        Parameters
        ----------
        ctx : Context, optional
            MCP context for logging and progress updates.
        """
        self.ctx = ctx
        self.logger = logging.getLogger(__name__)

    async def info(self, message: str) -> None:
        """Log info message to both MCP context and logger.

        Parameters
        ----------
        message : str
            Message to log.
        """
        self.logger.info(message)
        if self.ctx:
            await self.ctx.info(message)

    async def error(self, message: str) -> None:
        """Log error message to both MCP context and logger.

        Parameters
        ----------
        message : str
            Error message to log.
        """
        self.logger.error(message)
        if self.ctx:
            await self.ctx.error(message)

    async def warning(self, message: str) -> None:
        """Log warning message to both MCP context and logger.

        Parameters
        ----------
        message : str
            Warning message to log.
        """
        self.logger.warning(message)
        if self.ctx:
            await self.ctx.warning(message)

    def info_sync(self, message: str) -> None:
        """Log info message from sync context.

        Parameters
        ----------
        message : str
            Message to log.
        """
        self.logger.info(message)
        if self.ctx:
            anyio.from_thread.run(self.ctx.info, message)

    def error_sync(self, message: str) -> None:
        """Log error message from sync context.

        Parameters
        ----------
        message : str
            Error message to log.
        """
        self.logger.error(message)
        if self.ctx:
            anyio.from_thread.run(self.ctx.error, message)

    def warning_sync(self, message: str) -> None:
        """Log warning message from sync context.

        Parameters
        ----------
        message : str
            Warning message to log.
        """
        self.logger.warning(message)
        if self.ctx:
            anyio.from_thread.run(self.ctx.warning, message)


def setup_logging() -> logging.Logger:
    """Configure logging for the MCP server.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    config = get_config()
    logging.basicConfig(
        level=config.logging.level,
        format=config.logging.format
    )
    return logging.getLogger(config.logging.node_name)


def with_context_logging(func: F) -> F:
    """Decorator to add context logging to sync functions.

    Parameters
    ----------
    func : callable
        Function to decorate.

    Returns
    -------
    callable
        Decorated function with context logging.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Extract context from kwargs if present
        ctx = kwargs.get('ctx')
        context_manager = MCPContextManager(ctx)

        # Add context manager to kwargs
        kwargs['context_manager'] = context_manager

        try:
            return func(*args, **kwargs)
        except Nav2MCPError as e:
            error_msg = f"Navigation error in {func.__name__}: {e.message}"
            context_manager.error_sync(error_msg)
            return json.dumps(e.to_dict(), indent=2)
        except Exception as e:
            error_msg = f"Unexpected error in {func.__name__}: {str(e)}"
            context_manager.error_sync(error_msg)
            ros_error = ROSError(error_msg, e)
            return json.dumps(ros_error.to_dict(), indent=2)

    return wrapper  # type: ignore[return-value]


def with_nav2_active_check(func: F) -> F:
    """Decorator to ensure Nav2 is active before executing function.

    Parameters
    ----------
    func : callable
        Function to decorate.

    Returns
    -------
    callable
        Decorated function with Nav2 active check.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from .navigation import get_navigator

        navigator = get_navigator()
        context_manager = kwargs.get('context_manager')

        try:
            if context_manager:
                context_manager.info_sync(
                    'Waiting for Nav2 to become active...')

            navigator.waitUntilNav2Active()

            if context_manager:
                context_manager.info_sync('Nav2 is active')

            return func(*args, **kwargs)

        except Exception as e:
            if context_manager:
                context_manager.error_sync(f"Nav2 activation failed: {str(e)}")
            raise ROSError("Failed to activate Nav2", e)

    return wrapper  # type: ignore[return-value]


def create_success_message(operation: str, details: Dict[str, Any]) -> str:
    """Create a standardized success message.

    Parameters
    ----------
    operation : str
        The operation that succeeded.
    details : dict
        Additional details about the operation.

    Returns
    -------
    str
        Formatted success message.
    """
    detail_str = ", ".join([f"{k}={v}" for k, v in details.items()])
    return f"Successfully {operation}: {detail_str}"


def create_failure_message(
    operation: str, reason: str, details: Optional[Dict[str, Any]] = None
) -> str:
    """Create a standardized failure message.

    Parameters
    ----------
    operation : str
        The operation that failed.
    reason : str
        The reason for failure.
    details : dict, optional
        Additional details about the failure.

    Returns
    -------
    str
        Formatted failure message.
    """
    message = f"Failed to {operation}: {reason}"
    if details:
        detail_str = ", ".join([f"{k}={v}" for k, v in details.items()])
        message += f" ({detail_str})"
    return message


def validate_numeric_range(
    value: float,
    min_value: float,
    max_value: float,
    parameter_name: str
) -> None:
    """Validate that a numeric value is within specified range.

    Parameters
    ----------
    value : float
        Value to validate.
    min_value : float
        Minimum allowed value.
    max_value : float
        Maximum allowed value.
    parameter_name : str
        Name of the parameter for error messages.

    Raises
    ------
    ValueError
        If value is outside the specified range.
    """
    if not (min_value <= value <= max_value):
        raise ValueError(
            f"{parameter_name} must be between {min_value} and {max_value}, "
            f"got {value}"
        )


def safe_json_dumps(obj: Any, indent: int = 2) -> str:
    """Safely serialize object to JSON with error handling.

    Parameters
    ----------
    obj : Any
        Object to serialize.
    indent : int, optional
        JSON indentation (default: 2).

    Returns
    -------
    str
        JSON string or error message if serialization fails.
    """
    try:
        return json.dumps(obj, indent=indent, default=str)
    except (TypeError, ValueError) as e:
        return json.dumps({
            'error': 'JSON serialization failed',
            'message': str(e),
            'object_type': type(obj).__name__
        }, indent=indent)
