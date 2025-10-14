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


class TestParseWaypointsErrorCases:
    """Tests for parse_waypoints error handling."""

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_parse_waypoints_not_a_list(self, mock_navigator_class: Mock) -> None:
        """Test waypoint parsing with non-list JSON.

        Verifies that non-list JSON (like dict or string) raises appropriate error.
        """
        nav_manager = NavigationManager()

        # JSON string that parses to a dict, not a list
        with pytest.raises(NavigationError, match='Waypoints must be a list'):
            nav_manager.parse_waypoints('{"x": 1.0, "y": 2.0}')

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_parse_waypoints_exceeds_max(self, mock_navigator_class: Mock) -> None:
        """Test waypoint parsing with too many waypoints.

        Verifies that exceeding max_waypoints raises appropriate error.
        """
        nav_manager = NavigationManager()

        # Create a list with more than max_waypoints (default is 100)
        # Create 101 waypoints
        waypoints = [[i * 1.0, i * 2.0] for i in range(101)]
        waypoints_str = str(waypoints).replace("'", '"')

        with pytest.raises(NavigationError, match='Too many waypoints'):
            nav_manager.parse_waypoints(waypoints_str)

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_parse_waypoints_invalid_format(self, mock_navigator_class: Mock) -> None:
        """Test waypoint parsing with invalid waypoint format.

        Verifies that waypoints not in [x, y] format raise appropriate error.
        """
        nav_manager = NavigationManager()

        # Waypoint with 3 elements instead of 2
        with pytest.raises(NavigationError, match='must be \\[x, y\\] format'):
            nav_manager.parse_waypoints('[[1.0, 2.0, 3.0]]')

        # Waypoint as single number instead of list
        with pytest.raises(NavigationError, match='must be \\[x, y\\] format'):
            nav_manager.parse_waypoints('[1.0]')

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_parse_waypoints_invalid_coordinates(self, mock_navigator_class: Mock) -> None:
        """Test waypoint parsing with invalid coordinate values.

        Verifies that non-numeric coordinates raise appropriate error.
        """
        nav_manager = NavigationManager()

        # Waypoint with string values
        with pytest.raises(NavigationError, match='Invalid coordinates'):
            nav_manager.parse_waypoints('[["invalid", "coords"]]')


class TestCancelNavigationAlreadyComplete:
    """Tests for cancel_navigation when task is already complete."""

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_cancel_navigation_no_active_task(self, mock_navigator_class: Mock) -> None:
        """Test cancelling when no navigation task is active.

        Verifies that attempting to cancel when task is already complete
        returns appropriate message.
        """
        mock_navigator = Mock()
        mock_navigator.isTaskComplete.return_value = True
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        result = nav_manager.cancel_navigation(context_manager)

        assert 'No active navigation task' in result
        # cancelTask should not be called when task is already complete
        mock_navigator.cancelTask.assert_not_called()


class TestDockRobot:
    """Tests for dock_robot functionality."""

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_dock_robot_by_id(self, mock_navigator_class: Mock) -> None:
        """Test docking robot using dock ID.

        Verifies that docking by ID works correctly.
        """
        mock_navigator = Mock()
        mock_navigator.getResult.return_value = TaskResult.SUCCEEDED
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        result = nav_manager.dock_robot(
            dock_id='charging_dock_1',
            nav_to_dock=True,
            context_manager=context_manager
        )

        assert 'Successfully docked' in result
        assert 'dock ID: charging_dock_1' in result
        mock_navigator.dockRobotByID.assert_called_once_with('charging_dock_1', True)

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_dock_robot_by_pose(self, mock_navigator_class: Mock) -> None:
        """Test docking robot using pose.

        Verifies that docking by pose works correctly.
        """
        mock_navigator = Mock()
        mock_navigator.getResult.return_value = TaskResult.SUCCEEDED
        mock_clock = Mock()
        mock_time = Mock()
        mock_time.to_msg.return_value = Mock()
        mock_clock.now.return_value = mock_time
        mock_navigator.get_clock.return_value = mock_clock
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        # Create a dock pose
        dock_pose = nav_manager.create_pose_stamped(5.0, 3.0, 0.0)

        result = nav_manager.dock_robot(
            dock_pose=dock_pose,
            dock_type='nova_carter_dock',
            nav_to_dock=False,
            context_manager=context_manager
        )

        assert 'Successfully docked' in result
        mock_navigator.dockRobotByPose.assert_called_once_with(
            dock_pose, 'nova_carter_dock', False
        )

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_dock_robot_no_params(self, mock_navigator_class: Mock) -> None:
        """Test docking robot without dock_pose or dock_id.

        Verifies that ValueError is raised when neither is provided.
        """
        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        with pytest.raises(ValueError, match='Either dock_pose or dock_id must be provided'):
            nav_manager.dock_robot(context_manager=context_manager)

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_dock_robot_failure(self, mock_navigator_class: Mock) -> None:
        """Test docking robot failure.

        Verifies that docking failure raises NavigationError.
        """
        mock_navigator = Mock()
        mock_navigator.getResult.return_value = TaskResult.FAILED
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        with pytest.raises(NavigationError):
            nav_manager.dock_robot(
                dock_id='charging_dock_1',
                context_manager=context_manager
            )


class TestUndockRobot:
    """Tests for undock_robot functionality."""

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_undock_robot_success(self, mock_navigator_class: Mock) -> None:
        """Test successful robot undocking.

        Verifies that undocking works correctly.
        """
        mock_navigator = Mock()
        mock_navigator.getResult.return_value = TaskResult.SUCCEEDED
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        result = nav_manager.undock_robot(
            dock_type='nova_carter_dock',
            context_manager=context_manager
        )

        assert 'Successfully undocked' in result
        mock_navigator.undockRobot.assert_called_once_with('nova_carter_dock')

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_undock_robot_failure(self, mock_navigator_class: Mock) -> None:
        """Test undocking robot failure.

        Verifies that undocking failure raises NavigationError.
        """
        mock_navigator = Mock()
        mock_navigator.getResult.return_value = TaskResult.FAILED
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        with pytest.raises(NavigationError):
            nav_manager.undock_robot(
                dock_type='nova_carter_dock',
                context_manager=context_manager
            )


class TestGetPath:
    """Tests for get_path functionality."""

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_get_path_success(self, mock_navigator_class: Mock) -> None:
        """Test successful path computation.

        Verifies that get_path correctly computes a path.
        """
        mock_navigator = Mock()
        mock_clock = Mock()
        mock_time = Mock()
        mock_time.to_msg.return_value = Mock()
        mock_clock.now.return_value = mock_time
        mock_navigator.get_clock.return_value = mock_clock

        # Mock the path response - needs to be serializable
        mock_path = Mock()
        mock_path.poses = []
        # Make the mock serializable by patching safe_json_dumps
        mock_navigator.getPath.return_value = mock_path
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        with patch('nav2_mcp_server.utils.safe_json_dumps') as mock_dumps:
            mock_dumps.return_value = '{"path": "test"}'

            result = nav_manager.get_path(
                start_x=0.0,
                start_y=0.0,
                start_yaw=0.0,
                goal_x=5.0,
                goal_y=5.0,
                goal_yaw=1.57,
                planner_id='GridBased',
                use_start=True,
                context_manager=context_manager
            )

            assert result is not None
            assert result == '{"path": "test"}'
            mock_dumps.assert_called_once_with(mock_path)

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_get_path_without_start(self, mock_navigator_class: Mock) -> None:
        """Test path computation without explicit start pose.

        Verifies that get_path works when use_start=False.
        """
        mock_navigator = Mock()
        mock_clock = Mock()
        mock_time = Mock()
        mock_time.to_msg.return_value = Mock()
        mock_clock.now.return_value = mock_time
        mock_navigator.get_clock.return_value = mock_clock

        mock_path = Mock()
        mock_path.poses = []
        mock_navigator.getPath.return_value = mock_path
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()

        with patch('nav2_mcp_server.utils.safe_json_dumps') as mock_dumps:
            mock_dumps.return_value = '{"path": []}'

            result = nav_manager.get_path(
                start_x=0.0,
                start_y=0.0,
                start_yaw=0.0,
                goal_x=5.0,
                goal_y=5.0,
                goal_yaw=1.57,
                use_start=False
            )

            assert result is not None
            # Verify getPath was called (not getPathThroughPoses)
            mock_navigator.getPath.assert_called_once()


class TestLifecycleOperations:
    """Tests for lifecycle startup/shutdown operations."""

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_lifecycle_startup_success(self, mock_navigator_class: Mock) -> None:
        """Test successful Nav2 lifecycle startup.

        Verifies that lifecycle startup works correctly.
        """
        mock_navigator = Mock()
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        result = nav_manager.lifecycle_startup(context_manager)

        assert 'startup completed successfully' in result
        mock_navigator.lifecycleStartup.assert_called_once()

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_lifecycle_startup_failure(self, mock_navigator_class: Mock) -> None:
        """Test Nav2 lifecycle startup failure.

        Verifies that startup failure raises NavigationError.
        """
        mock_navigator = Mock()
        mock_navigator.lifecycleStartup.side_effect = Exception('Startup failed')
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        with pytest.raises(NavigationError, match='Failed to startup Nav2 lifecycle'):
            nav_manager.lifecycle_startup(context_manager)

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_lifecycle_shutdown_success(self, mock_navigator_class: Mock) -> None:
        """Test successful Nav2 lifecycle shutdown.

        Verifies that lifecycle shutdown works correctly.
        """
        mock_navigator = Mock()
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        result = nav_manager.lifecycle_shutdown(context_manager)

        assert 'shutdown completed successfully' in result
        mock_navigator.lifecycleShutdown.assert_called_once()

    @patch('nav2_mcp_server.navigation.BasicNavigator')
    def test_lifecycle_shutdown_failure(self, mock_navigator_class: Mock) -> None:
        """Test Nav2 lifecycle shutdown failure.

        Verifies that shutdown failure raises NavigationError.
        """
        mock_navigator = Mock()
        mock_navigator.lifecycleShutdown.side_effect = Exception('Shutdown failed')
        mock_navigator_class.return_value = mock_navigator

        nav_manager = NavigationManager()
        context_manager = MCPContextManager()

        with pytest.raises(NavigationError, match='Failed to shutdown Nav2 lifecycle'):
            nav_manager.lifecycle_shutdown(context_manager)
