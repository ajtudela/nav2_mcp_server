"""Tests for Nav2 MCP Server tools.

This module tests all tool implementations including error handling,
parameter validation, and correct data processing.
"""

import json
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest
from fastmcp import Client, FastMCP


class TestNavigateToPose:
    """Tests for navigate_to_pose tool."""

    @patch('nav2_mcp_server.tools.get_transform_manager')
    @patch('nav2_mcp_server.tools.get_navigation_manager')
    async def test_navigate_to_pose_success(
        self,
        mock_get_nav_manager: Mock,
        mock_get_tf_manager: Mock,
        test_server: FastMCP,
        mock_navigation_manager: Mock,
        sample_pose: Dict[str, Any]
    ) -> None:
        """Test successful navigation to pose."""
        mock_get_nav_manager.return_value = mock_navigation_manager
        async with Client(test_server) as client:
            result = await client.call_tool(
                'navigate_to_pose',
                {
                    'x': sample_pose['position']['x'],
                    'y': sample_pose['position']['y'],
                    'yaw': 1.57
                }
            )

            assert result.content
            mock_navigation_manager.navigate_to_pose.assert_called_once()

    @patch('nav2_mcp_server.tools.get_transform_manager')
    @patch('nav2_mcp_server.tools.get_navigation_manager')
    async def test_navigate_to_pose_invalid_coordinates(
        self,
        mock_get_nav_manager: Mock,
        mock_get_tf_manager: Mock,
        test_server: FastMCP
    ) -> None:
        """Test navigation with invalid coordinates."""
        async with Client(test_server) as client:
            with pytest.raises(Exception):
                await client.call_tool(
                    'navigate_to_pose',
                    {
                        'x': 'invalid',
                        'y': 2.0,
                        'yaw': 1.57
                    }
                )


class TestFollowWaypoints:
    """Tests for follow_waypoints tool."""

    @patch('nav2_mcp_server.tools.get_transform_manager')
    @patch('nav2_mcp_server.tools.get_navigation_manager')
    async def test_follow_waypoints_success(
        self,
        mock_get_nav_manager: Mock,
        mock_get_tf_manager: Mock,
        test_server: FastMCP,
        mock_navigation_manager: Mock
    ) -> None:
        """Test successful waypoint following."""
        waypoints_str = '[[1.0, 2.0], [3.0, 4.0]]'

        mock_get_nav_manager.return_value = mock_navigation_manager
        async with Client(test_server) as client:
            result = await client.call_tool(
                'follow_waypoints',
                {'waypoints': waypoints_str}
            )

            assert result.content
            mock_navigation_manager.follow_waypoints.assert_called_once()

    @patch('nav2_mcp_server.tools.get_transform_manager')
    @patch('nav2_mcp_server.tools.get_navigation_manager')
    async def test_follow_waypoints_empty_list(
        self,
        mock_get_nav_manager: Mock,
        mock_get_tf_manager: Mock,
        test_server: FastMCP,
        mock_navigation_manager: Mock
    ) -> None:
        """Test waypoint following with empty waypoint list."""
        mock_get_nav_manager.return_value = mock_navigation_manager
        async with Client(test_server) as client:
            result = await client.call_tool(
                'follow_waypoints',
                {'waypoints': '[]'}
            )

            assert result.content


class TestSpinRobot:
    """Tests for spin_robot tool."""

    @patch('nav2_mcp_server.tools.get_transform_manager')
    @patch('nav2_mcp_server.tools.get_navigation_manager')
    async def test_spin_robot_success(
        self,
        mock_get_nav_manager: Mock,
        mock_get_tf_manager: Mock,
        test_server: FastMCP,
        mock_navigation_manager: Mock
    ) -> None:
        """Test successful robot spinning."""
        mock_get_nav_manager.return_value = mock_navigation_manager
        async with Client(test_server) as client:
            result = await client.call_tool(
                'spin_robot',
                {'angle': 1.57}
            )

            assert result.content
            mock_navigation_manager.spin_robot.assert_called_once()

    @patch('nav2_mcp_server.tools.get_transform_manager')
    @patch('nav2_mcp_server.tools.get_navigation_manager')
    async def test_spin_robot_invalid_angle(
        self,
        mock_get_nav_manager: Mock,
        mock_get_tf_manager: Mock,
        test_server: FastMCP
    ) -> None:
        """Test robot spinning with invalid angle."""
        async with Client(test_server) as client:
            with pytest.raises(Exception):
                await client.call_tool(
                    'spin_robot',
                    {'angular_distance': 'invalid'}
                )


class TestBackupRobot:
    """Tests for backup_robot tool."""

    @patch('nav2_mcp_server.tools.get_transform_manager')
    @patch('nav2_mcp_server.tools.get_navigation_manager')
    async def test_backup_robot_success(
        self,
        mock_get_nav_manager: Mock,
        mock_get_tf_manager: Mock,
        test_server: FastMCP,
        mock_navigation_manager: Mock
    ) -> None:
        """Test successful robot backup."""
        mock_get_nav_manager.return_value = mock_navigation_manager
        async with Client(test_server) as client:
            result = await client.call_tool(
                'backup_robot',
                {
                    'distance': 1.0,
                    'speed': 0.15
                }
            )

            assert result.content
            mock_navigation_manager.backup_robot.assert_called_once()

    @patch('nav2_mcp_server.tools.get_transform_manager')
    @patch('nav2_mcp_server.tools.get_navigation_manager')
    async def test_backup_robot_null_speed(
        self,
        mock_get_nav_manager: Mock,
        mock_get_tf_manager: Mock,
        test_server: FastMCP,
        mock_navigation_manager: Mock
    ) -> None:
        """Test backup robot with None speed (should use default)."""
        mock_get_nav_manager.return_value = mock_navigation_manager
        async with Client(test_server) as client:
            result = await client.call_tool(
                'backup_robot',
                {
                    'distance': 1.0,
                    'speed': None
                }
            )

            assert result.content
            mock_navigation_manager.backup_robot.assert_called_once()

    @patch('nav2_mcp_server.tools.get_config')
    @patch('nav2_mcp_server.tools.get_transform_manager')
    @patch('nav2_mcp_server.tools.get_navigation_manager')
    async def test_backup_robot_default_speed(
        self,
        mock_get_nav_manager: Mock,
        mock_get_tf_manager: Mock,
        mock_get_config: Mock,
        test_server: FastMCP,
        mock_navigation_manager: Mock
    ) -> None:
        """Test robot backup with default speed (speed parameter = 0.0).

        Verifies that when speed is 0.0, the default backup speed from
        config is used.
        """
        mock_get_nav_manager.return_value = mock_navigation_manager
        # Set default backup speed in config
        config_obj = Mock()
        config_obj.navigation.default_backup_speed = 0.25
        mock_get_config.return_value = config_obj

        async with Client(test_server) as client:
            result = await client.call_tool(
                'backup_robot',
                {
                    'distance': 1.0
                    # speed not specified, should use default
                }
            )

            assert result.content
            mock_navigation_manager.backup_robot.assert_called_once()


class TestDockRobot:
    """Tests for dock_robot tool."""

    @patch('nav2_mcp_server.tools.get_transform_manager')
    @patch('nav2_mcp_server.tools.get_navigation_manager')
    async def test_dock_robot_success(
        self,
        mock_get_nav_manager: Mock,
        mock_get_tf_manager: Mock,
        test_server: FastMCP,
        mock_navigation_manager: Mock
    ) -> None:
        """Test successful robot docking."""
        mock_get_nav_manager.return_value = mock_navigation_manager
        async with Client(test_server) as client:
            result = await client.call_tool(
                'dock_robot',
                {
                    'x': 5.0,
                    'y': 2.0,
                    'yaw': 0.0
                }
            )

            assert result.content
            mock_navigation_manager.dock_robot.assert_called_once()


class TestUndockRobot:
    """Tests for undock_robot tool."""

    @patch('nav2_mcp_server.tools.get_transform_manager')
    @patch('nav2_mcp_server.tools.get_navigation_manager')
    async def test_undock_robot_success(
        self,
        mock_get_nav_manager: Mock,
        mock_get_tf_manager: Mock,
        test_server: FastMCP,
        mock_navigation_manager: Mock
    ) -> None:
        """Test successful robot undocking."""
        mock_get_nav_manager.return_value = mock_navigation_manager
        async with Client(test_server) as client:
            result = await client.call_tool(
                'undock_robot',
                {'dock_type': 'charging_dock'}
            )

            assert result.content
            mock_navigation_manager.undock_robot.assert_called_once()


class TestGetPath:
    """Tests for get_path tool."""

    @patch('nav2_mcp_server.tools.get_transform_manager')
    @patch('nav2_mcp_server.tools.get_navigation_manager')
    async def test_get_path_success(
        self,
        mock_get_nav_manager: Mock,
        mock_get_tf_manager: Mock,
        test_server: FastMCP,
        mock_navigation_manager: Mock,
        sample_path: Dict[str, Any]
    ) -> None:
        """Test successful path planning."""
        mock_navigation_manager.get_path.return_value = json.dumps(sample_path)
        mock_get_nav_manager.return_value = mock_navigation_manager

        async with Client(test_server) as client:
            result = await client.call_tool(
                'get_path',
                {
                    'start_x': 0.0,
                    'start_y': 0.0,
                    'start_yaw': 0.0,
                    'goal_x': 2.0,
                    'goal_y': 2.0,
                    'goal_yaw': 1.57,
                    'planner_id': 'GridBased'
                }
            )

            assert result.content
            mock_navigation_manager.get_path.assert_called_once()


class TestGetPathFromRobot:
    """Tests for get_path_from_robot tool."""

    @patch('nav2_mcp_server.tools.get_transform_manager')
    @patch('nav2_mcp_server.tools.get_navigation_manager')
    async def test_get_path_from_robot_success(
        self,
        mock_get_nav_manager: Mock,
        mock_get_tf_manager: Mock,
        test_server: FastMCP,
        mock_navigation_manager: Mock,
        sample_path: Dict[str, Any]
    ) -> None:
        """Test successful path planning from robot position."""
        # get_path_from_robot internally calls get_path
        mock_navigation_manager.get_path.return_value = json.dumps(sample_path)
        mock_get_nav_manager.return_value = mock_navigation_manager

        async with Client(test_server) as client:
            result = await client.call_tool(
                'get_path_from_robot',
                {
                    'goal_x': 2.0,
                    'goal_y': 2.0,
                    'goal_yaw': 1.57,
                    'planner_id': 'GridBased'
                }
            )

            assert result.content
            mock_navigation_manager.get_path.assert_called_once()


class TestClearCostmaps:
    """Tests for clear_costmaps tool."""

    @patch('nav2_mcp_server.tools.get_transform_manager')
    @patch('nav2_mcp_server.tools.get_navigation_manager')
    async def test_clear_costmaps_success(
        self,
        mock_get_nav_manager: Mock,
        mock_get_tf_manager: Mock,
        test_server: FastMCP,
        mock_navigation_manager: Mock
    ) -> None:
        """Test successful costmap clearing."""
        mock_get_nav_manager.return_value = mock_navigation_manager
        async with Client(test_server) as client:
            result = await client.call_tool(
                'clear_costmaps',
                {}
            )

            assert result.content
            mock_navigation_manager.clear_costmaps.assert_called_once()


class TestGetRobotPose:
    """Tests for get_robot_pose tool."""

    @patch('nav2_mcp_server.tools.get_transform_manager')
    @patch('nav2_mcp_server.tools.get_navigation_manager')
    async def test_get_robot_pose_success(
        self,
        mock_get_nav_manager: Mock,
        mock_get_tf_manager: Mock,
        test_server: FastMCP,
        mock_transform_manager: Mock
    ) -> None:
        """Test successful robot pose retrieval."""
        mock_get_tf_manager.return_value = mock_transform_manager
        async with Client(test_server) as client:
            result = await client.call_tool(
                'get_robot_pose',
                {}
            )

            assert result.content
            mock_transform_manager.get_robot_pose.assert_called_once()


class TestCancelNavigation:
    """Tests for cancel_navigation tool."""

    @patch('nav2_mcp_server.tools.get_transform_manager')
    @patch('nav2_mcp_server.tools.get_navigation_manager')
    async def test_cancel_navigation_success(
        self,
        mock_get_nav_manager: Mock,
        mock_get_tf_manager: Mock,
        test_server: FastMCP,
        mock_navigation_manager: Mock
    ) -> None:
        """Test successful navigation cancellation."""
        mock_get_nav_manager.return_value = mock_navigation_manager
        async with Client(test_server) as client:
            result = await client.call_tool(
                'cancel_navigation',
                {}
            )

            assert result.content
            mock_navigation_manager.cancel_navigation.assert_called_once()


class TestNav2Lifecycle:
    """Tests for nav2_lifecycle tool."""

    @patch('nav2_mcp_server.tools.get_transform_manager')
    @patch('nav2_mcp_server.tools.get_navigation_manager')
    async def test_nav2_lifecycle_startup(
        self,
        mock_get_nav_manager: Mock,
        mock_get_tf_manager: Mock,
        test_server: FastMCP,
        mock_navigation_manager: Mock,
        lifecycle_nodes: List[str]
    ) -> None:
        """Test successful Nav2 lifecycle startup."""
        mock_get_nav_manager.return_value = mock_navigation_manager
        async with Client(test_server) as client:
            result = await client.call_tool(
                'nav2_lifecycle',
                {'operation': 'startup'}
            )

            assert result.content
            mock_navigation_manager.lifecycle_startup.assert_called_once()

    @patch('nav2_mcp_server.tools.get_transform_manager')
    @patch('nav2_mcp_server.tools.get_navigation_manager')
    async def test_nav2_lifecycle_shutdown(
        self,
        mock_get_nav_manager: Mock,
        mock_get_tf_manager: Mock,
        test_server: FastMCP,
        mock_navigation_manager: Mock
    ) -> None:
        """Test successful Nav2 lifecycle shutdown."""
        mock_get_nav_manager.return_value = mock_navigation_manager
        async with Client(test_server) as client:
            result = await client.call_tool(
                'nav2_lifecycle',
                {'operation': 'shutdown'}
            )

            assert result.content
            mock_navigation_manager.lifecycle_shutdown.assert_called_once()

    @patch('nav2_mcp_server.tools.get_transform_manager')
    @patch('nav2_mcp_server.tools.get_navigation_manager')
    async def test_nav2_lifecycle_invalid_operation(
        self,
        mock_get_nav_manager: Mock,
        mock_get_tf_manager: Mock,
        test_server: FastMCP,
        mock_navigation_manager: Mock
    ) -> None:
        """Test Nav2 lifecycle with invalid operation.

        Verifies that an invalid operation parameter results in an error.
        """
        mock_get_nav_manager.return_value = mock_navigation_manager
        async with Client(test_server) as client:
            # Should handle invalid operation gracefully
            result = await client.call_tool(
                'nav2_lifecycle',
                {'operation': 'invalid_operation'}
            )

            # Result should contain error information
            assert result.content
            # The decorator should catch the ValueError and return JSON


class TestToolErrorHandling:
    """Tests for error handling across all tools."""

    @patch('nav2_mcp_server.tools.get_transform_manager')
    @patch('nav2_mcp_server.tools.get_navigation_manager')
    async def test_navigation_manager_exception(
        self,
        mock_get_nav_manager: Mock,
        mock_get_tf_manager: Mock,
        test_server: FastMCP
    ) -> None:
        """Test tool behavior when NavigationManager raises exception."""
        mock_nav_manager = Mock()
        mock_nav_manager.navigate_to_pose.side_effect = Exception(
            'Navigation system unavailable'
        )
        mock_get_nav_manager.return_value = mock_nav_manager

        async with Client(test_server) as client:
            # Should handle the exception and return error info
            result = await client.call_tool(
                'navigate_to_pose',
                {
                    'x': 1.0,
                    'y': 2.0,
                    'yaw': 1.57
                }
            )

            # Result should contain error information
            assert result.content
            assert 'error' in str(result.content).lower()

    @patch('nav2_mcp_server.tools.get_transform_manager')
    @patch('nav2_mcp_server.tools.get_navigation_manager')
    async def test_transform_manager_exception(
        self,
        mock_get_nav_manager: Mock,
        mock_get_tf_manager: Mock,
        test_server: FastMCP
    ) -> None:
        """Test tool behavior when TransformManager raises exception."""
        mock_tf_manager = Mock()
        mock_tf_manager.get_robot_pose.side_effect = Exception(
            'Transform lookup failed'
        )
        mock_get_tf_manager.return_value = mock_tf_manager

        async with Client(test_server) as client:
            # Should handle the exception and return error info
            result = await client.call_tool(
                'get_robot_pose',
                {}
            )

            # Result should contain error information
            assert result.content
            assert 'error' in str(result.content).lower()
