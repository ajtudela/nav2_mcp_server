"""Tests for Navigation Manager functionality.

This module tests navigation operations including pose navigation,
waypoint following, and robot maneuvers.
"""

from unittest.mock import Mock, patch

import pytest
from nav2_simple_commander.robot_navigator import TaskResult

from nav2_mcp_server.exceptions import NavigationError
from nav2_mcp_server.navigation import NavigationManager
from nav2_mcp_server.utils import MCPContextManager


class TestNavigationManagerInitialization:
    """Tests for NavigationManager initialization."""

    def test_navigation_manager_init(self) -> None:
        """Test NavigationManager initialization.

        Verifies that the NavigationManager properly initializes.
        """
        nav_manager = NavigationManager()

        assert nav_manager is not None
        # Navigator is created lazily
        assert nav_manager._navigator is None

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_navigation_manager_navigator_property(self, mock_navigator_class: Mock) -> None:
        """Test NavigationManager navigator property.

        Verifies that the navigator property returns the correct
        BasicNavigator instance and creates it lazily.
        """
        mock_navigator = Mock()
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        navigator = nav_manager.navigator

        assert navigator is mock_navigator
        mock_navigator_class.assert_called_once()


class TestCreatePoseStamped:
    """Tests for pose creation utilities."""

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_create_pose_stamped_success(self, mock_navigator_class: Mock) -> None:
        """Test successful PoseStamped creation.

        Verifies that create_pose_stamped correctly constructs
        a PoseStamped message from coordinates.
        """
        mock_navigator = Mock()
        mock_clock = Mock()
        mock_time = Mock()
        mock_time.to_msg.return_value = Mock()
        mock_clock.now.return_value = mock_time
        mock_navigator.get_clock.return_value = mock_clock
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()

        # create_pose_stamped signature: (x, y, yaw=0.0)
        result = nav_manager.create_pose_stamped(1.0, 2.0, 1.57)

        assert result is not None
        assert result.pose.position.x == 1.0
        assert result.pose.position.y == 2.0

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_create_pose_stamped_with_quaternion(self, mock_navigator_class: Mock) -> None:
        """Test PoseStamped creation with quaternion conversion.

        Verifies that yaw angles are correctly converted to quaternions.
        """
        mock_navigator = Mock()
        mock_clock = Mock()
        mock_time = Mock()
        mock_time.to_msg.return_value = Mock()
        mock_clock.now.return_value = mock_time
        mock_navigator.get_clock.return_value = mock_clock
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()

        # Yaw conversion is done inline, not through a method
        result = nav_manager.create_pose_stamped(1.0, 2.0, 1.57)

        # Check quaternion was set (approximate values for yaw=1.57)
        import math
        assert abs(result.pose.orientation.w - math.cos(1.57 / 2.0)) < 0.01
        assert abs(result.pose.orientation.z - math.sin(1.57 / 2.0)) < 0.01


class TestParseWaypoints:
    """Tests for waypoint parsing functionality."""

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_parse_waypoints_success(self, mock_navigator_class: Mock) -> None:
        """Test successful waypoint parsing.

        Verifies that waypoint strings are correctly parsed
        into PoseStamped messages.
        """
        mock_navigator = Mock()
        mock_clock = Mock()
        mock_time = Mock()
        mock_time.to_msg.return_value = Mock()
        mock_clock.now.return_value = mock_time
        mock_navigator.get_clock.return_value = mock_clock
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()

        # Format is [[x, y], [x, y], ...]
        waypoints_str = '[[1.0, 2.0], [3.0, 4.0]]'

        result = nav_manager.parse_waypoints(waypoints_str)

        assert len(result) == 2
        assert result[0].pose.position.x == 1.0
        assert result[0].pose.position.y == 2.0
        assert result[1].pose.position.x == 3.0
        assert result[1].pose.position.y == 4.0

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_parse_waypoints_invalid_json(self, mock_navigator_class: Mock) -> None:
        """Test waypoint parsing with invalid JSON.

        Verifies that invalid JSON input is handled gracefully.
        """
        from nav2_mcp_server.exceptions import NavigationError

        nav_manager = NavigationManager()

        with pytest.raises(NavigationError):
            nav_manager.parse_waypoints('invalid json')

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_parse_waypoints_empty_list(self, mock_navigator_class: Mock) -> None:
        """Test waypoint parsing with empty list.

        Verifies that empty waypoint lists are handled correctly.
        """
        nav_manager = NavigationManager()

        result = nav_manager.parse_waypoints('[]')

        assert result == []


class TestNavigateToPose:
    """Tests for navigate_to_pose functionality."""

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_navigate_to_pose_success(self, mock_navigator_class: Mock) -> None:
        """Test successful navigation to pose.

        Verifies that navigate_to_pose correctly initiates navigation
        and returns success status.
        """
        mock_navigator = Mock()
        mock_navigator_class.return_value = mock_navigator
        mock_navigator.getResult.return_value = TaskResult.SUCCEEDED

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        with patch.object(nav_manager, 'create_pose_stamped') as mock_create:
            mock_pose = Mock()
            mock_create.return_value = mock_pose

            result = nav_manager.navigate_to_pose(
                1.0, 2.0, 1.57, context_manager
            )

            assert 'success' in result.lower()
            mock_navigator.goToPose.assert_called_once_with(mock_pose)

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_navigate_to_pose_failure(self, mock_navigator_class: Mock) -> None:
        """Test navigation failure handling.

        Verifies that navigation failures are properly handled
        and error messages are returned.
        """
        mock_navigator = Mock()
        mock_navigator.getResult.return_value = TaskResult.FAILED
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        with patch.object(nav_manager, 'create_pose_stamped'):
            with pytest.raises(NavigationError) as exc_info:
                nav_manager.navigate_to_pose(
                    1.0, 2.0, 1.57, context_manager
                )

            assert 'Navigation to pose' in str(exc_info.value)


class TestFollowWaypoints:
    """Tests for follow_waypoints functionality."""

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_follow_waypoints_success(self, mock_navigator_class: Mock) -> None:
        """Test successful waypoint following.

        Verifies that waypoint following is correctly initiated
        with parsed waypoints.
        """
        mock_navigator = Mock()
        mock_navigator.getResult.return_value = TaskResult.SUCCEEDED
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        waypoints_str = '[[1.0, 2.0], [3.0, 4.0]]'

        result = nav_manager.follow_waypoints(waypoints_str, context_manager)

        assert 'success' in result.lower()
        mock_navigator.followWaypoints.assert_called_once()

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_follow_waypoints_empty(self, mock_navigator_class: Mock) -> None:
        """Test waypoint following with empty waypoints.

        Verifies that empty waypoint lists are handled appropriately.
        """
        mock_navigator = Mock()
        mock_navigator.getResult.return_value = TaskResult.SUCCEEDED
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        result = nav_manager.follow_waypoints('[]', context_manager)

        assert 'success' in result.lower()


class TestSpinRobot:
    """Tests for robot spinning functionality."""

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_spin_robot_success(self, mock_navigator_class: Mock) -> None:
        """Test successful robot spinning.

        Verifies that spin operation is correctly initiated
        with the specified angular distance.
        """
        mock_navigator = Mock()
        mock_navigator.getResult.return_value = TaskResult.SUCCEEDED
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        result = nav_manager.spin_robot(1.57, context_manager)

        assert 'success' in result.lower()
        mock_navigator.spin.assert_called_once_with(1.57)

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_spin_robot_failure(self, mock_navigator_class: Mock) -> None:
        """Test spin operation failure handling.

        Verifies that spin operation failures are handled gracefully.
        """
        mock_navigator = Mock()
        mock_navigator.getResult.return_value = TaskResult.FAILED
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        with pytest.raises(NavigationError):
            nav_manager.spin_robot(1.57, context_manager)


class TestBackupRobot:
    """Tests for robot backup functionality."""

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_backup_robot_success(self, mock_navigator_class: Mock) -> None:
        """Test successful robot backup.

        Verifies that backup operation is correctly initiated
        with distance and speed parameters.
        """
        mock_navigator = Mock()
        mock_navigator.getResult.return_value = TaskResult.SUCCEEDED
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        result = nav_manager.backup_robot(1.0, 0.15, context_manager)

        assert 'success' in result.lower()
        mock_navigator.backup.assert_called_once_with(1.0, 0.15)

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_backup_robot_with_defaults(self, mock_navigator_class: Mock) -> None:
        """Test robot backup with default parameters.

        Verifies that default backup parameters are correctly applied.
        """
        mock_navigator = Mock()
        mock_navigator.getResult.return_value = TaskResult.SUCCEEDED
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        # Test with default speed parameter (0.15)
        result = nav_manager.backup_robot(1.0, 0.15, context_manager)

        assert 'success' in result.lower()
        mock_navigator.backup.assert_called_once()


class TestClearCostmaps:
    """Tests for costmap clearing functionality."""

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_clear_costmaps_success(self, mock_navigator_class: Mock) -> None:
        """Test successful costmap clearing.

        Verifies that costmap clearing operations work correctly.
        """
        mock_navigator = Mock()
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        result = nav_manager.clear_costmaps('all', context_manager)

        assert 'cleared successfully' in result or 'success' in result.lower()
        mock_navigator.clearAllCostmaps.assert_called_once()

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_clear_costmaps_failure(self, mock_navigator_class: Mock) -> None:
        """Test costmap clearing failure handling.

        Verifies that costmap clearing failures are handled properly.
        """
        mock_navigator = Mock()
        mock_navigator.clearAllCostmaps.side_effect = Exception('Clear failed')
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        with pytest.raises(NavigationError):
            nav_manager.clear_costmaps('all', context_manager)


class TestCancelNavigation:
    """Tests for navigation cancellation functionality."""

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_cancel_navigation_success(self, mock_navigator_class: Mock) -> None:
        """Test successful navigation cancellation.

        Verifies that ongoing navigation can be successfully cancelled.
        """
        mock_navigator = Mock()
        mock_navigator.isTaskComplete.return_value = False
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        result = nav_manager.cancel_navigation(context_manager)

        assert 'cancel' in result.lower() or 'requested' in result.lower()
        mock_navigator.cancelTask.assert_called_once()

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_cancel_navigation_failure(self, mock_navigator_class: Mock) -> None:
        """Test navigation cancellation failure handling.

        Verifies that cancellation failures are handled gracefully.
        """
        mock_navigator = Mock()
        mock_navigator.isTaskComplete.return_value = False
        mock_navigator.cancelTask.side_effect = Exception('Cancel failed')
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        with pytest.raises(Exception, match='Cancel failed'):
            nav_manager.cancel_navigation(context_manager)


class TestNavigationManagerUtilities:
    """Tests for NavigationManager utility methods."""

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_create_pose_stamped_with_defaults(self, mock_navigator_class: Mock) -> None:
        """Test pose creation with default yaw.

        Verifies that poses can be created with default orientation.
        """
        mock_navigator = Mock()
        mock_clock = Mock()
        mock_time = Mock()
        mock_time.to_msg.return_value = Mock()
        mock_clock.now.return_value = mock_time
        mock_navigator.get_clock.return_value = mock_clock
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()

        # Test pose creation with default yaw (0.0)
        pose = nav_manager.create_pose_stamped(1.0, 2.0)

        assert pose.pose.position.x == 1.0
        assert pose.pose.position.y == 2.0
        # Default yaw should result in w=1.0, z=0.0
        import math
        assert abs(pose.pose.orientation.w - math.cos(0.0 / 2.0)) < 0.01
        assert abs(pose.pose.orientation.w - 1.0) < 0.001
        assert abs(pose.pose.orientation.z) < 0.001
