"""Tests for NavigationManager operations.

This module tests the NavigationManager class and its navigation
operations including pose navigation, waypoint following, and
action management.
"""

from unittest.mock import Mock, patch

import pytest

from nav2_mcp_server.navigation import NavigationManager
from nav2_mcp_server.utils import MCPContextManager


class TestNavigationManagerInitialization:
    """Tests for NavigationManager initialization."""

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_navigation_manager_init(self, mock_navigator_class):
        """Test NavigationManager initialization.

        Verifies that the NavigationManager properly initializes
        with a BasicNavigator instance.
        """
        mock_navigator = Mock()
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()

        assert nav_manager is not None
        mock_navigator_class.assert_called_once()

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_navigation_manager_navigator_property(self, mock_navigator_class):
        """Test NavigationManager navigator property.

        Verifies that the navigator property returns the correct
        BasicNavigator instance.
        """
        mock_navigator = Mock()
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        navigator = nav_manager.navigator

        assert navigator is mock_navigator


class TestCreatePoseStamped:
    """Tests for pose creation utilities."""

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_create_pose_stamped_success(self, mock_navigator_class):
        """Test successful PoseStamped creation.

        Verifies that create_pose_stamped correctly constructs
        a PoseStamped message from coordinates.
        """
        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        with patch('nav2_mcp_server.navigation.PoseStamped') as mock_pose:
            mock_pose_instance = Mock()
            mock_pose.return_value = mock_pose_instance

            result = nav_manager.create_pose_stamped(
                1.0, 2.0, 1.57, 'map', context_manager
            )

            assert result is mock_pose_instance
            mock_pose.assert_called_once()

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_create_pose_stamped_with_quaternion(self, mock_navigator_class):
        """Test PoseStamped creation with quaternion conversion.

        Verifies that yaw angles are correctly converted to quaternions.
        """
        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        with patch('nav2_mcp_server.navigation.PoseStamped'):
            with patch.object(nav_manager, 'yaw_to_quaternion') as mock_quat:
                mock_quat.return_value = {'x': 0, 'y': 0, 'z': 0.707, 'w': 0.707}

                nav_manager.create_pose_stamped(
                    1.0, 2.0, 1.57, 'map', context_manager
                )

                mock_quat.assert_called_once_with(1.57)


class TestParseWaypoints:
    """Tests for waypoint parsing functionality."""

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_parse_waypoints_success(self, mock_navigator_class):
        """Test successful waypoint parsing.

        Verifies that waypoint strings are correctly parsed
        into PoseStamped messages.
        """
        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        waypoints_str = '[{"position": {"x": 1.0, "y": 2.0, "z": 0.0}, "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}]'

        with patch.object(nav_manager, 'create_pose_stamped') as mock_create:
            mock_pose = Mock()
            mock_create.return_value = mock_pose

            result = nav_manager.parse_waypoints(waypoints_str, context_manager)

            assert len(result) == 1
            assert result[0] is mock_pose
            mock_create.assert_called_once()

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_parse_waypoints_invalid_json(self, mock_navigator_class):
        """Test waypoint parsing with invalid JSON.

        Verifies that invalid JSON input is handled gracefully.
        """
        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        with pytest.raises(ValueError):
            nav_manager.parse_waypoints('invalid json', context_manager)

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_parse_waypoints_empty_list(self, mock_navigator_class):
        """Test waypoint parsing with empty list.

        Verifies that empty waypoint lists are handled correctly.
        """
        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        result = nav_manager.parse_waypoints('[]', context_manager)

        assert result == []


class TestNavigateToPose:
    """Tests for navigate_to_pose functionality."""

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_navigate_to_pose_success(self, mock_navigator_class):
        """Test successful navigation to pose.

        Verifies that navigate_to_pose correctly initiates navigation
        and returns success status.
        """
        mock_navigator = Mock()
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        with patch.object(nav_manager, 'create_pose_stamped') as mock_create:
            mock_pose = Mock()
            mock_create.return_value = mock_pose

            result = nav_manager.navigate_to_pose(
                1.0, 2.0, 1.57, 'map', context_manager
            )

            assert 'started successfully' in result
            mock_navigator.goToPose.assert_called_once_with(mock_pose)

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_navigate_to_pose_failure(self, mock_navigator_class):
        """Test navigation failure handling.

        Verifies that navigation failures are properly handled
        and error messages are returned.
        """
        mock_navigator = Mock()
        mock_navigator.goToPose.side_effect = Exception('Navigation failed')
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        with patch.object(nav_manager, 'create_pose_stamped'):
            result = nav_manager.navigate_to_pose(
                1.0, 2.0, 1.57, 'map', context_manager
            )

            assert 'error' in result or 'failed' in result


class TestFollowWaypoints:
    """Tests for follow_waypoints functionality."""

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_follow_waypoints_success(self, mock_navigator_class):
        """Test successful waypoint following.

        Verifies that waypoint following is correctly initiated
        with parsed waypoints.
        """
        mock_navigator = Mock()
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        waypoints_str = '[{"position": {"x": 1.0, "y": 2.0, "z": 0.0}}]'

        with patch.object(nav_manager, 'parse_waypoints') as mock_parse:
            mock_waypoints = [Mock(), Mock()]
            mock_parse.return_value = mock_waypoints

            result = nav_manager.follow_waypoints(waypoints_str, context_manager)

            assert 'started successfully' in result
            mock_navigator.followWaypoints.assert_called_once_with(mock_waypoints)

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_follow_waypoints_empty(self, mock_navigator_class):
        """Test waypoint following with empty waypoints.

        Verifies that empty waypoint lists are handled appropriately.
        """
        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        with patch.object(nav_manager, 'parse_waypoints') as mock_parse:
            mock_parse.return_value = []

            result = nav_manager.follow_waypoints('[]', context_manager)

            assert 'no waypoints' in result.lower() or 'empty' in result.lower()


class TestSpinRobot:
    """Tests for robot spinning functionality."""

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_spin_robot_success(self, mock_navigator_class):
        """Test successful robot spinning.

        Verifies that spin operation is correctly initiated
        with the specified angular distance.
        """
        mock_navigator = Mock()
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        result = nav_manager.spin_robot(1.57, context_manager)

        assert 'started successfully' in result
        mock_navigator.spin.assert_called_once_with(1.57)

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_spin_robot_failure(self, mock_navigator_class):
        """Test spin operation failure handling.

        Verifies that spin operation failures are handled gracefully.
        """
        mock_navigator = Mock()
        mock_navigator.spin.side_effect = Exception('Spin failed')
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        result = nav_manager.spin_robot(1.57, context_manager)

        assert 'error' in result or 'failed' in result


class TestBackupRobot:
    """Tests for robot backup functionality."""

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_backup_robot_success(self, mock_navigator_class):
        """Test successful robot backup.

        Verifies that backup operation is correctly initiated
        with distance and speed parameters.
        """
        mock_navigator = Mock()
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        result = nav_manager.backup_robot(1.0, 0.15, context_manager)

        assert 'started successfully' in result
        mock_navigator.backup.assert_called_once_with(1.0, 0.15)

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_backup_robot_with_defaults(self, mock_navigator_class):
        """Test robot backup with default parameters.

        Verifies that default backup parameters are correctly applied.
        """
        mock_navigator = Mock()
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        # Test with default speed parameter
        result = nav_manager.backup_robot(1.0, None, context_manager)

        assert 'started successfully' in result
        # Should use default speed if None provided
        mock_navigator.backup.assert_called_once()


class TestClearCostmaps:
    """Tests for costmap clearing functionality."""

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_clear_costmaps_success(self, mock_navigator_class):
        """Test successful costmap clearing.

        Verifies that costmap clearing operations work correctly.
        """
        mock_navigator = Mock()
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        result = nav_manager.clear_costmaps(context_manager)

        assert 'cleared successfully' in result
        mock_navigator.clearAllCostmaps.assert_called_once()

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_clear_costmaps_failure(self, mock_navigator_class):
        """Test costmap clearing failure handling.

        Verifies that costmap clearing failures are handled properly.
        """
        mock_navigator = Mock()
        mock_navigator.clearAllCostmaps.side_effect = Exception('Clear failed')
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        result = nav_manager.clear_costmaps(context_manager)

        assert 'error' in result or 'failed' in result


class TestCancelNavigation:
    """Tests for navigation cancellation functionality."""

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_cancel_navigation_success(self, mock_navigator_class):
        """Test successful navigation cancellation.

        Verifies that ongoing navigation can be successfully cancelled.
        """
        mock_navigator = Mock()
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        result = nav_manager.cancel_navigation(context_manager)

        assert 'cancelled successfully' in result
        mock_navigator.cancelTask.assert_called_once()

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_cancel_navigation_failure(self, mock_navigator_class):
        """Test navigation cancellation failure handling.

        Verifies that cancellation failures are handled gracefully.
        """
        mock_navigator = Mock()
        mock_navigator.cancelTask.side_effect = Exception('Cancel failed')
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        result = nav_manager.cancel_navigation(context_manager)

        assert 'error' in result or 'failed' in result


class TestNavigationManagerUtilities:
    """Tests for NavigationManager utility methods."""

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_yaw_to_quaternion_conversion(self, mock_navigator_class):
        """Test yaw to quaternion conversion utility.

        Verifies that yaw angles are correctly converted to quaternions.
        """
        nav_manager = NavigationManager()

        # Test conversion of common angles
        quat_0 = nav_manager.yaw_to_quaternion(0.0)
        quat_90 = nav_manager.yaw_to_quaternion(1.5708)  # 90 degrees
        quat_180 = nav_manager.yaw_to_quaternion(3.14159)  # 180 degrees

        # Verify quaternion structure
        assert 'x' in quat_0 and 'y' in quat_0 and 'z' in quat_0 and 'w' in quat_0
        assert 'x' in quat_90 and 'y' in quat_90 and 'z' in quat_90 and 'w' in quat_90
        assert 'x' in quat_180 and 'y' in quat_180 and 'z' in quat_180 and 'w' in quat_180

        # For 0 degrees, w should be close to 1.0
        assert abs(quat_0['w'] - 1.0) < 0.001
        assert abs(quat_0['z']) < 0.001
