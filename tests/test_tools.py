"""Tests for Nav2 MCP Server tools.

This module tests all tool implementations including error handling,
parameter validation, and correct data processing.
"""

from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest
from fastmcp import Client

from nav2_mcp_server.server import create_server


class TestNavigateToPose:
    """Tests for navigate_to_pose tool."""

    async def test_navigate_to_pose_success(
        self,
        mock_navigation_manager: Mock,
        sample_pose: Dict[str, Any]
    ):
        """Test successful navigation to pose.

        Verifies that the tool correctly initiates navigation
        when provided with valid pose parameters.
        """
        with patch('nav2_mcp_server.server.get_config'):
            with patch(
                'nav2_mcp_server.tools.get_navigation_manager',
                return_value=mock_navigation_manager
            ):
                with patch('nav2_mcp_server.tools.get_transform_manager'):
                    with patch(
                        'nav2_mcp_server.resources.get_transform_manager'
                    ):
                        server = create_server()
                        async with Client(server) as client:
                            result = await client.call_tool(
                                'navigate_to_pose',
                                {
                                    'x': sample_pose['position']['x'],
                                    'y': sample_pose['position']['y'],
                                    'theta': 1.57,
                                    'frame_id': 'map'
                                }
                            )

                            assert result.content
                            mock_navigation_manager.navigate_to_pose\
                                .assert_called_once()

    async def test_navigate_to_pose_invalid_coordinates(self):
        """Test navigation with invalid coordinates.

        Verifies that the tool handles invalid coordinate inputs gracefully.
        """
        with patch('nav2_mcp_server.server.get_config'):
            with patch('nav2_mcp_server.tools.get_navigation_manager'):
                with patch('nav2_mcp_server.tools.get_transform_manager'):
                    with patch(
                        'nav2_mcp_server.resources.get_transform_manager'
                    ):
                        server = create_server()
                        async with Client(server) as client:
                            with pytest.raises(Exception):
                                await client.call_tool(
                                    'navigate_to_pose',
                                    {
                                        'x': 'invalid',
                                        'y': 2.0,
                                        'theta': 1.57,
                                        'frame_id': 'map'
                                    }
                                )


class TestFollowWaypoints:
    """Tests for follow_waypoints tool."""

    async def test_follow_waypoints_success(
        self,
        mock_navigation_manager: Mock,
        sample_waypoints: List[Dict[str, Any]]
    ):
        """Test successful waypoint following.

        Verifies that the tool correctly initiates waypoint following
        when provided with valid waypoint data.
        """
        waypoints_str = str(sample_waypoints)
        
        with patch('nav2_mcp_server.server.get_config'):
            with patch(
                'nav2_mcp_server.tools.get_navigation_manager',
                return_value=mock_navigation_manager
            ):
                with patch('nav2_mcp_server.tools.get_transform_manager'):
                    with patch(
                        'nav2_mcp_server.resources.get_transform_manager'
                    ):
                        server = create_server()
                        async with Client(server) as client:
                            result = await client.call_tool(
                                'follow_waypoints',
                                {'waypoints': waypoints_str}
                            )

                            assert result.content
                            mock_navigation_manager.follow_waypoints\
                                .assert_called_once()

    async def test_follow_waypoints_empty_list(self):
        """Test waypoint following with empty waypoint list.

        Verifies that the tool handles empty waypoint inputs appropriately.
        """
        with patch('nav2_mcp_server.server.get_config'):
            with patch('nav2_mcp_server.tools.get_navigation_manager'):
                with patch('nav2_mcp_server.tools.get_transform_manager'):
                    with patch(
                        'nav2_mcp_server.resources.get_transform_manager'
                    ):
                        server = create_server()
                        async with Client(server) as client:
                            result = await client.call_tool(
                                'follow_waypoints',
                                {'waypoints': '[]'}
                            )
                            
                            # Should handle empty waypoints gracefully
                            assert result.content


class TestSpinRobot:
    """Tests for spin_robot tool."""

    async def test_spin_robot_success(
        self,
        mock_navigation_manager: Mock
    ):
        """Test successful robot spinning.

        Verifies that the tool correctly initiates robot spinning
        with the specified angular distance.
        """
        with patch('nav2_mcp_server.server.get_config'):
            with patch(
                'nav2_mcp_server.tools.get_navigation_manager',
                return_value=mock_navigation_manager
            ):
                with patch('nav2_mcp_server.tools.get_transform_manager'):
                    with patch(
                        'nav2_mcp_server.resources.get_transform_manager'
                    ):
                        server = create_server()
                        async with Client(server) as client:
                            result = await client.call_tool(
                                'spin_robot',
                                {'angular_distance': 1.57}
                            )

                            assert result.content
                            mock_navigation_manager.spin_robot\
                                .assert_called_once()

    async def test_spin_robot_invalid_angle(self):
        """Test robot spinning with invalid angle.

        Verifies that the tool handles invalid angular distance inputs.
        """
        with patch('nav2_mcp_server.server.get_config'):
            with patch('nav2_mcp_server.tools.get_navigation_manager'):
                with patch('nav2_mcp_server.tools.get_transform_manager'):
                    with patch(
                        'nav2_mcp_server.resources.get_transform_manager'
                    ):
                        server = create_server()
                        async with Client(server) as client:
                            with pytest.raises(Exception):
                                await client.call_tool(
                                    'spin_robot',
                                    {'angular_distance': 'invalid'}
                                )


class TestBackupRobot:
    """Tests for backup_robot tool."""

    async def test_backup_robot_success(
        self,
        mock_navigation_manager: Mock
    ):
        """Test successful robot backup.

        Verifies that the tool correctly initiates robot backup
        with the specified distance and speed.
        """
        with patch('nav2_mcp_server.server.get_config'):
            with patch(
                'nav2_mcp_server.tools.get_navigation_manager',
                return_value=mock_navigation_manager
            ):
                with patch('nav2_mcp_server.tools.get_transform_manager'):
                    with patch(
                        'nav2_mcp_server.resources.get_transform_manager'
                    ):
                        server = create_server()
                        async with Client(server) as client:
                            result = await client.call_tool(
                                'backup_robot',
                                {
                                    'backup_distance': 1.0,
                                    'backup_speed': 0.15
                                }
                            )

                            assert result.content
                            mock_navigation_manager.backup_robot\
                                .assert_called_once()


class TestDockRobot:
    """Tests for dock_robot tool."""

    async def test_dock_robot_success(
        self,
        mock_navigation_manager: Mock
    ):
        """Test successful robot docking.

        Verifies that the tool correctly initiates robot docking
        with optional dock parameters.
        """
        with patch('nav2_mcp_server.server.get_config'):
            with patch(
                'nav2_mcp_server.tools.get_navigation_manager',
                return_value=mock_navigation_manager
            ):
                with patch('nav2_mcp_server.tools.get_transform_manager'):
                    with patch(
                        'nav2_mcp_server.resources.get_transform_manager'
                    ):
                        server = create_server()
                        async with Client(server) as client:
                            result = await client.call_tool(
                                'dock_robot',
                                {'dock_id': 'dock_1'}
                            )

                            assert result.content
                            mock_navigation_manager.dock_robot.assert_called_once()


class TestUndockRobot:
    """Tests for undock_robot tool."""

    async def test_undock_robot_success(
        self,
        mock_navigation_manager: Mock
    ):
        """Test successful robot undocking.

        Verifies that the tool correctly initiates robot undocking.
        """
        with patch('nav2_mcp_server.server.get_config'):
            with patch(
                'nav2_mcp_server.tools.get_navigation_manager',
                return_value=mock_navigation_manager
            ):
                with patch('nav2_mcp_server.tools.get_transform_manager'):
                    with patch(
                        'nav2_mcp_server.resources.get_transform_manager'
                    ):
                        server = create_server()
                        async with Client(server) as client:
                            result = await client.call_tool(
                                'undock_robot',
                                {'dock_type': 'charging_dock'}
                            )

                            assert result.content
                            mock_navigation_manager.undock_robot.assert_called_once()


class TestGetPath:
    """Tests for get_path tool."""

    async def test_get_path_success(
        self,
        mock_navigation_manager: Mock,
        sample_path: Dict[str, Any]
    ):
        """Test successful path planning.

        Verifies that the tool correctly computes a path between
        start and goal poses.
        """
        mock_navigation_manager.get_path.return_value = sample_path
        
        with patch('nav2_mcp_server.server.get_config'):
            with patch(
                'nav2_mcp_server.tools.get_navigation_manager',
                return_value=mock_navigation_manager
            ):
                with patch('nav2_mcp_server.tools.get_transform_manager'):
                    with patch(
                        'nav2_mcp_server.resources.get_transform_manager'
                    ):
                        server = create_server()
                        async with Client(server) as client:
                            result = await client.call_tool(
                                'get_path',
                                {
                                    'start_x': 0.0,
                                    'start_y': 0.0,
                                    'start_theta': 0.0,
                                    'goal_x': 2.0,
                                    'goal_y': 2.0,
                                    'goal_theta': 1.57,
                                    'planner_id': 'GridBased'
                                }
                            )

                            assert result.content
                            mock_navigation_manager.get_path.assert_called_once()


class TestGetPathFromRobot:
    """Tests for get_path_from_robot tool."""

    async def test_get_path_from_robot_success(
        self,
        mock_navigation_manager: Mock,
        sample_path: Dict[str, Any]
    ):
        """Test successful path planning from robot position.

        Verifies that the tool correctly computes a path from
        the current robot position to a goal.
        """
        mock_navigation_manager.get_path_from_robot.return_value = sample_path
        
        with patch('nav2_mcp_server.server.get_config'):
            with patch(
                'nav2_mcp_server.tools.get_navigation_manager',
                return_value=mock_navigation_manager
            ):
                with patch('nav2_mcp_server.tools.get_transform_manager'):
                    with patch(
                        'nav2_mcp_server.resources.get_transform_manager'
                    ):
                        server = create_server()
                        async with Client(server) as client:
                            result = await client.call_tool(
                                'get_path_from_robot',
                                {
                                    'goal_x': 2.0,
                                    'goal_y': 2.0,
                                    'goal_theta': 1.57,
                                    'planner_id': 'GridBased'
                                }
                            )

                            assert result.content
                            mock_navigation_manager.get_path_from_robot.assert_called_once()


class TestClearCostmaps:
    """Tests for clear_costmaps tool."""

    async def test_clear_costmaps_success(
        self,
        mock_navigation_manager: Mock
    ):
        """Test successful costmap clearing.

        Verifies that the tool correctly clears the costmaps.
        """
        with patch('nav2_mcp_server.server.get_config'):
            with patch(
                'nav2_mcp_server.tools.get_navigation_manager',
                return_value=mock_navigation_manager
            ):
                with patch('nav2_mcp_server.tools.get_transform_manager'):
                    with patch(
                        'nav2_mcp_server.resources.get_transform_manager'
                    ):
                        server = create_server()
                        async with Client(server) as client:
                            result = await client.call_tool(
                                'clear_costmaps',
                                {}
                            )

                            assert result.content
                            mock_navigation_manager.clear_costmaps.assert_called_once()


class TestGetRobotPose:
    """Tests for get_robot_pose tool."""

    async def test_get_robot_pose_success(
        self,
        mock_transform_manager: Mock
    ):
        """Test successful robot pose retrieval.

        Verifies that the tool correctly retrieves the current robot pose.
        """
        with patch('nav2_mcp_server.server.get_config'):
            with patch('nav2_mcp_server.tools.get_navigation_manager'):
                with patch(
                    'nav2_mcp_server.tools.get_transform_manager',
                    return_value=mock_transform_manager
                ):
                    with patch(
                        'nav2_mcp_server.resources.get_transform_manager'
                    ):
                        server = create_server()
                        async with Client(server) as client:
                            result = await client.call_tool(
                                'get_robot_pose',
                                {}
                            )

                            assert result.content
                            mock_transform_manager.get_robot_pose.assert_called_once()


class TestCancelNavigation:
    """Tests for cancel_navigation tool."""

    async def test_cancel_navigation_success(
        self,
        mock_navigation_manager: Mock
    ):
        """Test successful navigation cancellation.

        Verifies that the tool correctly cancels ongoing navigation.
        """
        with patch('nav2_mcp_server.server.get_config'):
            with patch(
                'nav2_mcp_server.tools.get_navigation_manager',
                return_value=mock_navigation_manager
            ):
                with patch('nav2_mcp_server.tools.get_transform_manager'):
                    with patch(
                        'nav2_mcp_server.resources.get_transform_manager'
                    ):
                        server = create_server()
                        async with Client(server) as client:
                            result = await client.call_tool(
                                'cancel_navigation',
                                {}
                            )

                            assert result.content
                            mock_navigation_manager.cancel_navigation.assert_called_once()


class TestNav2Lifecycle:
    """Tests for nav2_lifecycle tool."""

    async def test_nav2_lifecycle_startup(
        self,
        mock_navigation_manager: Mock,
        lifecycle_nodes: List[str]
    ):
        """Test successful Nav2 lifecycle startup.

        Verifies that the tool correctly manages Nav2 node lifecycle.
        """
        with patch('nav2_mcp_server.server.get_config'):
            with patch(
                'nav2_mcp_server.tools.get_navigation_manager',
                return_value=mock_navigation_manager
            ):
                with patch('nav2_mcp_server.tools.get_transform_manager'):
                    with patch(
                        'nav2_mcp_server.resources.get_transform_manager'
                    ):
                        server = create_server()
                        async with Client(server) as client:
                            result = await client.call_tool(
                                'nav2_lifecycle',
                                {'action': 'startup'}
                            )

                            assert result.content
                            mock_navigation_manager.nav2_lifecycle.assert_called_once()

    async def test_nav2_lifecycle_shutdown(
        self,
        mock_navigation_manager: Mock
    ):
        """Test successful Nav2 lifecycle shutdown.

        Verifies that the tool correctly shuts down Nav2 nodes.
        """
        with patch('nav2_mcp_server.server.get_config'):
            with patch(
                'nav2_mcp_server.tools.get_navigation_manager',
                return_value=mock_navigation_manager
            ):
                with patch('nav2_mcp_server.tools.get_transform_manager'):
                    with patch(
                        'nav2_mcp_server.resources.get_transform_manager'
                    ):
                        server = create_server()
                        async with Client(server) as client:
                            result = await client.call_tool(
                                'nav2_lifecycle',
                                {'action': 'shutdown'}
                            )

                            assert result.content
                            mock_navigation_manager.nav2_lifecycle.assert_called_once()


class TestToolErrorHandling:
    """Tests for error handling across all tools."""

    async def test_navigation_manager_exception(self):
        """Test tool behavior when NavigationManager raises exception.

        Verifies that tools handle NavigationManager exceptions gracefully.
        """
        mock_nav_manager = Mock()
        mock_nav_manager.navigate_to_pose.side_effect = Exception(
            'Navigation system unavailable'
        )
        
        with patch('nav2_mcp_server.server.get_config'):
            with patch(
                'nav2_mcp_server.tools.get_navigation_manager',
                return_value=mock_nav_manager
            ):
                with patch('nav2_mcp_server.tools.get_transform_manager'):
                    with patch(
                        'nav2_mcp_server.resources.get_transform_manager'
                    ):
                        server = create_server()
                        async with Client(server) as client:
                            # Should handle the exception and return error info
                            result = await client.call_tool(
                                'navigate_to_pose',
                                {
                                    'x': 1.0,
                                    'y': 2.0,
                                    'theta': 1.57,
                                    'frame_id': 'map'
                                }
                            )
                            
                            # Result should contain error information
                            assert result.content
                            assert 'error' in str(result.content).lower()

    async def test_transform_manager_exception(self):
        """Test tool behavior when TransformManager raises exception.

        Verifies that tools handle TransformManager exceptions gracefully.
        """
        mock_tf_manager = Mock()
        mock_tf_manager.get_robot_pose.side_effect = Exception(
            'Transform lookup failed'
        )
        
        with patch('nav2_mcp_server.server.get_config'):
            with patch('nav2_mcp_server.tools.get_navigation_manager'):
                with patch(
                    'nav2_mcp_server.tools.get_transform_manager',
                    return_value=mock_tf_manager
                ):
                    with patch(
                        'nav2_mcp_server.resources.get_transform_manager'
                    ):
                        server = create_server()
                        async with Client(server) as client:
                            # Should handle the exception and return error info
                            result = await client.call_tool(
                                'get_robot_pose',
                                {}
                            )
                            
                            # Result should contain error information
                            assert result.content
                            assert 'error' in str(result.content).lower()