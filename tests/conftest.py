"""Shared pytest fixtures for Nav2 MCP Server tests.

This module provides reusable fixtures for testing the Nav2 MCP server,
including mock ROS2 interfaces, sample navigation data, and server
configurations.
"""

from typing import Any, Dict, List
from unittest.mock import Mock
import math

import pytest
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer

from nav2_mcp_server.server import create_server
from nav2_mcp_server.navigation import NavigationManager
from nav2_mcp_server.transforms import TransformManager
from nav2_mcp_server.utils import MCPContextManager


@pytest.fixture
def mock_ros2_initialized():
    """Mock ROS2 initialization."""
    with pytest.mock.patch('rclpy.init'):
        with pytest.mock.patch('rclpy.shutdown'):
            yield


@pytest.fixture
def mock_navigation_manager() -> Mock:
    """Provide a mock NavigationManager for testing.

    Returns
    -------
    Mock
        Mock NavigationManager with preset return values and side effects.
    """
    mock_nav_manager = Mock(spec=NavigationManager)
    
    # Mock successful navigation result
    mock_nav_manager.navigate_to_pose.return_value = (
        "Navigation to pose started successfully"
    )
    mock_nav_manager.follow_waypoints.return_value = (
        "Waypoint following started successfully"
    )
    mock_nav_manager.spin_robot.return_value = (
        "Spin operation started successfully"
    )
    mock_nav_manager.backup_robot.return_value = (
        "Backup operation started successfully"
    )
    mock_nav_manager.dock_robot.return_value = (
        "Docking operation started successfully"
    )
    mock_nav_manager.undock_robot.return_value = (
        "Undocking operation started successfully"
    )
    mock_nav_manager.clear_costmaps.return_value = (
        "Costmaps cleared successfully"
    )
    mock_nav_manager.cancel_navigation.return_value = (
        "Navigation cancelled successfully"
    )
    mock_nav_manager.nav2_lifecycle.return_value = (
        "Lifecycle operation completed successfully"
    )
    
    # Mock path planning results
    mock_nav_manager.get_path.return_value = {
        'path': [
            {'x': 0.0, 'y': 0.0, 'theta': 0.0},
            {'x': 1.0, 'y': 1.0, 'theta': 0.785},
            {'x': 2.0, 'y': 2.0, 'theta': 1.571}
        ],
        'length': 2.828,
        'status': 'success'
    }
    
    mock_nav_manager.get_path_from_robot.return_value = {
        'path': [
            {'x': 1.0, 'y': 1.0, 'theta': 0.0},
            {'x': 2.0, 'y': 2.0, 'theta': 0.785}
        ],
        'length': 1.414,
        'status': 'success'
    }
    
    return mock_nav_manager


@pytest.fixture
def mock_transform_manager() -> Mock:
    """Provide a mock TransformManager for testing.

    Returns
    -------
    Mock
        Mock TransformManager with preset return values.
    """
    mock_tf_manager = Mock(spec=TransformManager)
    
    # Mock robot pose
    mock_tf_manager.get_robot_pose.return_value = {
        'pose': {
            'position': {'x': 1.0, 'y': 2.0, 'z': 0.0},
            'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
        },
        'header': {
            'frame_id': 'map',
            'stamp': {'sec': 1640995200, 'nanosec': 0}
        },
        'yaw': 0.0,
        'status': 'success'
    }
    
    return mock_tf_manager


@pytest.fixture
def mock_context_manager() -> Mock:
    """Provide a mock MCPContextManager for testing.

    Returns
    -------
    Mock
        Mock MCPContextManager for context handling.
    """
    mock_context = Mock(spec=MCPContextManager)
    mock_context.info.return_value = None
    mock_context.warning.return_value = None
    mock_context.error.return_value = None
    return mock_context


@pytest.fixture
def sample_pose() -> Dict[str, Any]:
    """Provide a sample pose for testing.

    Returns
    -------
    Dict[str, Any]
        Sample pose dictionary.
    """
    return {
        'position': {'x': 1.0, 'y': 2.0, 'z': 0.0},
        'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.707, 'w': 0.707}
    }


@pytest.fixture
def sample_waypoints() -> List[Dict[str, Any]]:
    """Provide sample waypoints for testing.

    Returns
    -------
    List[Dict[str, Any]]
        List of waypoint dictionaries.
    """
    return [
        {
            'position': {'x': 1.0, 'y': 1.0, 'z': 0.0},
            'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
        },
        {
            'position': {'x': 2.0, 'y': 2.0, 'z': 0.0},
            'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.707, 'w': 0.707}
        },
        {
            'position': {'x': 3.0, 'y': 1.0, 'z': 0.0},
            'orientation': {'x': 0.0, 'y': 0.0, 'z': 1.0, 'w': 0.0}
        }
    ]


@pytest.fixture
def sample_path() -> Dict[str, Any]:
    """Provide a sample path for testing.

    Returns
    -------
    Dict[str, Any]
        Sample path dictionary with poses and metadata.
    """
    return {
        'path': [
            {'x': 0.0, 'y': 0.0, 'theta': 0.0},
            {'x': 0.5, 'y': 0.5, 'theta': 0.785},
            {'x': 1.0, 'y': 1.0, 'theta': 1.571},
            {'x': 1.5, 'y': 1.0, 'theta': 3.14},
            {'x': 2.0, 'y': 1.0, 'theta': 0.0}
        ],
        'length': 3.414,
        'status': 'success',
        'planning_time': 0.15
    }


@pytest.fixture
def mock_ros_pose_stamped() -> Mock:
    """Provide a mock ROS PoseStamped message.

    Returns
    -------
    Mock
        Mock PoseStamped message with realistic structure.
    """
    pose_stamped = Mock(spec=PoseStamped)
    pose_stamped.header.frame_id = 'map'
    pose_stamped.header.stamp.sec = 1640995200
    pose_stamped.header.stamp.nanosec = 0
    
    pose_stamped.pose.position.x = 1.0
    pose_stamped.pose.position.y = 2.0
    pose_stamped.pose.position.z = 0.0
    
    pose_stamped.pose.orientation.x = 0.0
    pose_stamped.pose.orientation.y = 0.0
    pose_stamped.pose.orientation.z = 0.0
    pose_stamped.pose.orientation.w = 1.0
    
    return pose_stamped


@pytest.fixture
def mock_tf_buffer() -> Mock:
    """Provide a mock TF2 Buffer for testing.

    Returns
    -------
    Mock
        Mock TF2 Buffer with transform lookup capabilities.
    """
    mock_buffer = Mock(spec=Buffer)
    
    # Mock transform lookup
    mock_transform = Mock()
    mock_transform.transform.translation.x = 1.0
    mock_transform.transform.translation.y = 2.0
    mock_transform.transform.translation.z = 0.0
    mock_transform.transform.rotation.x = 0.0
    mock_transform.transform.rotation.y = 0.0
    mock_transform.transform.rotation.z = 0.0
    mock_transform.transform.rotation.w = 1.0
    mock_transform.header.frame_id = 'map'
    mock_transform.child_frame_id = 'base_link'
    
    mock_buffer.lookup_transform.return_value = mock_transform
    
    return mock_buffer


@pytest.fixture
def test_server():
    """Provide the FastMCP server instance for testing.

    Returns
    -------
    FastMCP
        The configured MCP server instance.
    """
    return create_server()


# Navigation result fixtures
@pytest.fixture
def navigation_success_result() -> Dict[str, Any]:
    """Success result for navigation operations."""
    return {
        'status': 'success',
        'message': 'Navigation completed successfully',
        'result_code': 4,  # SUCCEEDED
        'duration': 15.5
    }


@pytest.fixture
def navigation_failure_result() -> Dict[str, Any]:
    """Failure result for navigation operations."""
    return {
        'status': 'failed',
        'message': 'Navigation failed: Goal unreachable',
        'result_code': 5,  # ABORTED
        'duration': 8.2
    }


@pytest.fixture
def lifecycle_nodes() -> List[str]:
    """List of Nav2 lifecycle nodes."""
    return [
        'controller_server',
        'smoother_server',
        'planner_server',
        'behavior_server',
        'bt_navigator',
        'waypoint_follower',
        'velocity_smoother'
    ]


# Error scenarios fixtures
@pytest.fixture
def ros_connection_error() -> Exception:
    """ROS connection error for testing."""
    return ConnectionError("Failed to connect to ROS2 nodes")


@pytest.fixture
def navigation_timeout_error() -> Exception:
    """Navigation timeout error for testing."""
    return TimeoutError("Navigation action timed out")


@pytest.fixture
def transform_lookup_error() -> Exception:
    """Transform lookup error for testing."""
    return Exception("Transform lookup failed: Frame 'odom' does not exist")


# Math utilities for testing
def create_quaternion_from_yaw(yaw: float) -> Dict[str, float]:
    """Create quaternion from yaw angle.

    Parameters
    ----------
    yaw : float
        Yaw angle in radians.

    Returns
    -------
    Dict[str, float]
        Quaternion representation.
    """
    return {
        'x': 0.0,
        'y': 0.0,
        'z': math.sin(yaw / 2.0),
        'w': math.cos(yaw / 2.0)
    }