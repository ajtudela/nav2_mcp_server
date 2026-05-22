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

"""Navigation management for Nav2 MCP Server.

This module provides classes and functions for managing Nav2 navigation
operations including pose navigation, waypoint following, and robot control.
"""

import json
import math
from typing import Any, List, Optional

from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.duration import Duration

from .config import get_config
from .exceptions import create_navigation_error_from_result, NavigationError, NavigationErrorCode
from .utils import MCPContextManager, validate_numeric_range


class NavigationManager:
    """Manages Nav2 navigation operations and state."""

    def __init__(self) -> None:
        """Initialize the navigation manager."""
        self._navigator: Optional[BasicNavigator] = None
        self.config = get_config()

    @property
    def navigator(self) -> BasicNavigator:
        """Get or create the navigator instance.

        Returns
        -------
        BasicNavigator
            The Nav2 navigator instance.
        """
        if self._navigator is None:
            self._navigator = BasicNavigator()
        return self._navigator

    def create_pose_stamped(
        self, x: float, y: float, yaw: float = 0.0
    ) -> PoseStamped:
        """Create a PoseStamped message from coordinates and orientation.

        Parameters
        ----------
        x : float
            X coordinate in map frame.
        y : float
            Y coordinate in map frame.
        yaw : float, optional
            Orientation in radians (default: 0.0).

        Returns
        -------
        PoseStamped
            The pose message.
        """
        pose = PoseStamped()
        pose.header.frame_id = self.config.navigation.map_frame
        pose.header.stamp = self.navigator.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y

        # Convert yaw to quaternion
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        pose.pose.orientation.z = math.sin(yaw / 2.0)

        return pose

    def parse_waypoints(self, waypoints_str: str) -> List[PoseStamped]:
        """Parse waypoints from JSON string to PoseStamped list.

        Parameters
        ----------
        waypoints_str : str
            JSON string with waypoint coordinates.

        Returns
        -------
        List[PoseStamped]
            List of pose messages.

        Raises
        ------
        NavigationError
            If waypoints format is invalid.
        """
        try:
            waypoints_data = json.loads(waypoints_str)
        except json.JSONDecodeError as e:
            raise NavigationError(
                f'Invalid waypoints JSON format: {e}',
                NavigationErrorCode.INVALID_WAYPOINTS,
                {'json_error': str(e)}
            )

        if not isinstance(waypoints_data, list):
            raise NavigationError(
                'Waypoints must be a list of [x, y] coordinates',
                NavigationErrorCode.INVALID_WAYPOINTS,
                {'received_type': type(waypoints_data).__name__}
            )

        if len(waypoints_data) > self.config.navigation.max_waypoints:
            raise NavigationError(
                f'Too many waypoints. Maximum: '
                f'{self.config.navigation.max_waypoints}',
                NavigationErrorCode.INVALID_WAYPOINTS,
                {'waypoint_count': len(waypoints_data)}
            )

        poses = []
        for i, waypoint in enumerate(waypoints_data):
            if not isinstance(waypoint, list) or len(waypoint) != 2:
                raise NavigationError(
                    f'Waypoint {i} must be [x, y] format',
                    NavigationErrorCode.INVALID_WAYPOINTS,
                    {'waypoint_index': i, 'waypoint_data': waypoint}
                )

            try:
                x, y = float(waypoint[0]), float(waypoint[1])
                poses.append(self.create_pose_stamped(x, y))
            except (ValueError, TypeError) as e:
                raise NavigationError(
                    f'Invalid coordinates in waypoint {i}: {e}',
                    NavigationErrorCode.INVALID_WAYPOINTS,
                    {'waypoint_index': i, 'error': str(e)}
                )

        return poses

    def navigate_to_pose(
        self,
        x: float,
        y: float,
        yaw: float,
        context_manager: MCPContextManager
    ) -> str:
        """Navigate to a specific pose.

        Parameters
        ----------
        x : float
            X coordinate of target pose.
        y : float
            Y coordinate of target pose.
        yaw : float
            Target orientation in radians.
        context_manager : MCPContextManager
            Context manager for logging.

        Returns
        -------
        str
            Success message with pose details.

        Raises
        ------
        NavigationError
            If navigation fails.
        """
        goal_pose = self.create_pose_stamped(x, y, yaw)

        self.navigator.get_logger().info(
            f'Navigating to pose: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}'
        )

        self._ensure_action_server(
            self.navigator.nav_to_pose_client, '/navigate_to_pose'
        )
        self.navigator.goToPose(goal_pose)
        self._monitor_navigation_progress(context_manager, 'pose navigation')

        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            return (f'Successfully navigated to pose '
                    f'({x:.2f}, {y:.2f}, {yaw:.2f})')
        else:
            raise create_navigation_error_from_result(
                result, 'Navigation to pose')

    def follow_waypoints(
        self, waypoints_str: str, context_manager: MCPContextManager
    ) -> str:
        """Follow a sequence of waypoints.

        Parameters
        ----------
        waypoints_str : str
            JSON string with waypoint coordinates.
        context_manager : MCPContextManager
            Context manager for logging.

        Returns
        -------
        str
            Success message with waypoint count.

        Raises
        ------
        NavigationError
            If waypoint following fails.
        """
        poses = self.parse_waypoints(waypoints_str)

        self.navigator.get_logger().info(f'Following {len(poses)} waypoints')
        context_manager.info_sync(
            f'Starting waypoint following with {len(poses)} points'
        )

        self._ensure_action_server(
            self.navigator.follow_waypoints_client, '/follow_waypoints'
        )
        self.navigator.followWaypoints(poses)
        self._monitor_waypoint_progress(context_manager)

        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            return (f'Successfully completed waypoint following '
                    f'with {len(poses)} waypoints')
        else:
            raise create_navigation_error_from_result(
                result, 'Waypoint following')

    def spin_robot(
        self, angle: float, context_manager: MCPContextManager
    ) -> str:
        """Spin robot by specified angle.

        Parameters
        ----------
        angle : float
            Angle to spin in radians.
        context_manager : MCPContextManager
            Context manager for logging.

        Returns
        -------
        str
            Success message with angle.

        Raises
        ------
        NavigationError
            If spin operation fails.
        """
        self.navigator.get_logger().info(
            f'Spinning robot by {angle:.2f} radians')
        context_manager.info_sync(
            f'Starting spin operation: {angle:.2f} radians')

        self._ensure_action_server(self.navigator.spin_client, '/spin')
        self.navigator.spin(angle)
        self._monitor_navigation_progress(context_manager, 'spin operation')

        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            return f'Successfully spun robot by {angle:.2f} radians'
        else:
            raise create_navigation_error_from_result(result, 'Spin operation')

    def backup_robot(
        self, distance: float, speed: float, context_manager: MCPContextManager
    ) -> str:
        """Back up robot by specified distance.

        Parameters
        ----------
        distance : float
            Distance to back up in meters.
        speed : float
            Backup speed in m/s.
        context_manager : MCPContextManager
            Context manager for logging.

        Returns
        -------
        str
            Success message with distance.

        Raises
        ------
        NavigationError
            If backup operation fails.
        ValueError
            If distance or speed parameters are invalid.
        """
        # Validate parameters
        validate_numeric_range(
            distance,
            self.config.navigation.min_backup_distance,
            self.config.navigation.max_backup_distance,
            'distance'
        )
        validate_numeric_range(
            speed,
            self.config.navigation.min_backup_speed,
            self.config.navigation.max_backup_speed,
            'speed'
        )

        self.navigator.get_logger().info(
            f'Backing up robot: {distance:.2f}m at {speed:.2f}m/s'
        )
        context_manager.info_sync(
            f'Starting backup: {distance:.2f}m at {speed:.2f}m/s'
        )

        self._ensure_action_server(self.navigator.backup_client, '/backup')
        self.navigator.backup(distance, speed)
        self._monitor_navigation_progress(context_manager, 'backup operation')

        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            return f'Successfully backed up {distance:.2f} meters'
        else:
            raise create_navigation_error_from_result(
                result, 'Backup operation')

    def drive_on_heading(
        self, distance: float, speed: float, context_manager: MCPContextManager
    ) -> str:
        """Drive robot forward by specified distance on its current heading.

        Parameters
        ----------
        distance : float
            Distance to drive forward in meters (positive value).
        speed : float
            Forward speed in m/s.
        context_manager : MCPContextManager
            Context manager for logging.

        Returns
        -------
        str
            Success message with distance.

        Raises
        ------
        NavigationError
            If drive operation fails.
        ValueError
            If distance or speed parameters are invalid.
        """
        # Reuse backup limits (physical motion limits, direction-agnostic)
        validate_numeric_range(
            distance,
            self.config.navigation.min_backup_distance,
            self.config.navigation.max_backup_distance,
            'distance'
        )
        validate_numeric_range(
            speed,
            self.config.navigation.min_backup_speed,
            self.config.navigation.max_backup_speed,
            'speed'
        )

        time_allowance = int(distance / speed + 5)

        self.navigator.get_logger().info(
            f'Driving robot on heading: {distance:.2f}m at {speed:.2f}m/s'
        )
        context_manager.info_sync(
            f'Starting drive on heading: {distance:.2f}m at {speed:.2f}m/s'
        )

        self._ensure_action_server(
            self.navigator.drive_on_heading_client, '/drive_on_heading'
        )
        self.navigator.driveOnHeading(distance, speed, time_allowance)
        self._monitor_navigation_progress(
            context_manager, 'drive on heading operation'
        )

        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            return f'Successfully drove {distance:.2f} meters on heading'
        else:
            raise create_navigation_error_from_result(
                result, 'Drive on heading operation')

    def approach_target(
        self,
        target_x_base: float,
        target_y_base: float,
        standoff_m: float,
        speed: float,
        context_manager: MCPContextManager,
    ) -> str:
        """Drive the robot to ``standoff_m`` from a target given in base_footprint.

        Given a target's xy in base_footprint frame, spin to face the target
        if the bearing is significant, then drive forward by
        ``dist - standoff_m`` along the now-aligned heading using
        ``drive_on_heading``. Returns immediately if the target is already
        within ``standoff_m + 0.10m``.

        The caller is responsible for any post-approach perception check.
        This method does not re-verify the target position after driving.

        Parameters
        ----------
        target_x_base : float
            Target x in the robot's base_footprint frame (forward = +x).
        target_y_base : float
            Target y in the robot's base_footprint frame (left = +y).
        standoff_m : float
            Desired distance from the target after the approach.
        speed : float
            Forward drive speed in m/s.
        context_manager : MCPContextManager
            Context manager for logging.

        Returns
        -------
        str
            Summary of what was executed.
        """
        target_dist = math.hypot(target_x_base, target_y_base)
        bearing = math.atan2(target_y_base, target_x_base)

        self.navigator.get_logger().info(
            f'approach_target: base=({target_x_base:.2f},{target_y_base:.2f}) '
            f'dist={target_dist:.2f}m bearing={math.degrees(bearing):.1f}deg '
            f'standoff={standoff_m:.2f}m'
        )
        context_manager.info_sync(
            f'approach_target: dist={target_dist:.2f}m -> standoff={standoff_m:.2f}m'
        )

        # Already within standoff (with 10cm tolerance). Don't drive — that
        # would back away from a target the manipulator can already reach.
        if target_dist <= standoff_m + 0.10:
            return (
                f'Already at {target_dist:.2f}m (<= standoff '
                f'{standoff_m:.2f}m + 0.10m); no approach needed'
            )

        # Spin to face the target if the bearing is significant. ~8 deg
        # threshold avoids burning a spin call on near-aligned targets.
        if abs(bearing) > math.radians(8):
            self.spin_robot(bearing, context_manager)

        # Drive forward by (dist - standoff) along the now-aligned heading.
        # Range validation, action-server check, and result handling all
        # live in drive_on_heading.
        drive_dist = max(target_dist - standoff_m, 0.20)
        self.drive_on_heading(drive_dist, speed, context_manager)

        return (
            f'Approached target: spun {math.degrees(bearing):.1f}deg, '
            f'drove {drive_dist:.2f}m -> ~{standoff_m:.2f}m standoff'
        )

    def clear_costmaps(
        self, costmap_type: str, context_manager: MCPContextManager
    ) -> str:
        """Clear navigation costmaps.

        Parameters
        ----------
        costmap_type : str
            Type of costmap to clear: 'global', 'local', or 'all'.
        context_manager : MCPContextManager
            Context manager for logging.

        Returns
        -------
        str
            Success message.

        Raises
        ------
        NavigationError
            If costmap clearing fails.
        """
        valid_types = {'global', 'local', 'all'}
        if costmap_type not in valid_types:
            raise NavigationError(
                f"Invalid costmap type '{costmap_type}'. "
                f"Valid types: {', '.join(valid_types)}",
                NavigationErrorCode.INVALID_PARAMETERS,
                {'valid_types': list(valid_types), 'received': costmap_type}
            )

        context_manager.info_sync(f'Clearing {costmap_type} costmap(s)...')

        try:
            if costmap_type == 'global':
                self.navigator.clearGlobalCostmap()
                message = 'Global costmap cleared successfully'
            elif costmap_type == 'local':
                self.navigator.clearLocalCostmap()
                message = 'Local costmap cleared successfully'
            else:  # costmap_type == 'all'
                self.navigator.clearAllCostmaps()
                message = 'All costmaps cleared successfully'

            context_manager.info_sync(message)
            return message

        except Exception as e:
            raise NavigationError(
                f'Failed to clear {costmap_type} costmap(s): {str(e)}',
                NavigationErrorCode.ROS_ERROR,
                {'costmap_type': costmap_type, 'error': str(e)}
            )

    def cancel_navigation(self, context_manager: MCPContextManager) -> str:
        """Cancel current navigation task.

        Parameters
        ----------
        context_manager : MCPContextManager
            Context manager for logging.

        Returns
        -------
        str
            Status message.
        """
        context_manager.info_sync('Canceling current navigation task...')

        if self.navigator.isTaskComplete():
            message = 'No active navigation task to cancel'
        else:
            self.navigator.cancelTask()
            message = 'Navigation task cancellation requested'

        context_manager.info_sync(message)
        return message

    def lifecycle_startup(self, context_manager: MCPContextManager) -> str:
        """Startup Nav2 lifecycle nodes.

        Parameters
        ----------
        context_manager : MCPContextManager
            Context manager for logging.

        Returns
        -------
        str
            Success message.

        Raises
        ------
        NavigationError
            If lifecycle startup fails.
        """
        context_manager.info_sync('Performing Nav2 startup...')

        try:
            self.navigator.lifecycleStartup()
            message = 'Nav2 lifecycle startup completed successfully'
            context_manager.info_sync(message)
            return message
        except Exception as e:
            raise NavigationError(
                f'Failed to startup Nav2 lifecycle: {str(e)}',
                NavigationErrorCode.ROS_ERROR,
                {'error': str(e)}
            )

    def lifecycle_shutdown(self, context_manager: MCPContextManager) -> str:
        """Shutdown Nav2 lifecycle nodes.

        Parameters
        ----------
        context_manager : MCPContextManager
            Context manager for logging.

        Returns
        -------
        str
            Success message.

        Raises
        ------
        NavigationError
            If lifecycle shutdown fails.
        """
        context_manager.info_sync('Performing Nav2 shutdown...')

        try:
            self.navigator.lifecycleShutdown()
            message = 'Nav2 lifecycle shutdown completed successfully'
            context_manager.info_sync(message)
            return message
        except Exception as e:
            raise NavigationError(
                f'Failed to shutdown Nav2 lifecycle: {str(e)}',
                NavigationErrorCode.ROS_ERROR,
                {'error': str(e)}
            )

    def get_path(
        self,
        start_x: float,
        start_y: float,
        start_yaw: float,
        goal_x: float,
        goal_y: float,
        goal_yaw: float,
        planner_id: str = '',
        use_start: bool = True,
        context_manager: Optional[MCPContextManager] = None
    ) -> str:
        """
        Compute a navigation path between two poses (start and goal).

        Parameters
        ----------
        start_x : float
            X coordinate of start pose.
        start_y : float
            Y coordinate of start pose.
        start_yaw : float
            Orientation of start pose in radians.
        goal_x : float
            X coordinate of goal pose.
        goal_y : float
            Y coordinate of goal pose.
        goal_yaw : float
            Orientation of goal pose in radians.
        planner_id : str, optional
            Planner ID to use (default: '').
        use_start : bool, default=True
            Whether to include the start pose in the path.
        context_manager : MCPContextManager, optional
            Context manager for logging.

        Returns
        -------
        str
            JSON string with the computed path.
        """
        start_pose = self.create_pose_stamped(start_x, start_y, start_yaw)
        goal_pose = self.create_pose_stamped(goal_x, goal_y, goal_yaw)
        path = self.navigator.getPath(
            start_pose, goal_pose, planner_id, use_start
        )
        # If context_manager is provided, log the event
        if context_manager:
            context_manager.info_sync(
                f'Computed path from ({start_x}, {start_y}, {start_yaw}) to '
                f'({goal_x}, {goal_y}, {goal_yaw})'
            )
        # Serialize the path to safe JSON
        from .utils import safe_json_dumps
        return safe_json_dumps(path)

    def dock_robot(
        self,
        dock_pose: Optional[PoseStamped] = None,
        dock_id: str = '',
        dock_type: str = '',
        nav_to_dock: bool = True,
        context_manager: Optional[MCPContextManager] = None
    ) -> str:
        """Dock the robot to a charging station or dock.

        Parameters
        ----------
        dock_pose : PoseStamped, optional
            Pose of the dock to navigate to. If None, dock_id must be provided.
        dock_id : str, optional
            ID of the dock to use. If empty, dock_pose must be provided.
        dock_type : str, optional
            Type of dock or empty to use the default.
        nav_to_dock : bool, default=True
            Whether to navigate to the staging pose before docking.
        context_manager : MCPContextManager
            Context manager for logging.

        Returns
        -------
        str
            Success message with docking details.

        Raises
        ------
        NavigationError
            If docking operation fails.
        ValueError
            If neither dock_pose nor dock_id is provided.
        """
        if dock_pose is None and not dock_id:
            raise ValueError('Either dock_pose or dock_id must be provided')

        if dock_pose is not None:
            # Dock using pose
            self.navigator.get_logger().info(
                f'Docking robot at pose: x={dock_pose.pose.position.x:.2f}, '
                f'y={dock_pose.pose.position.y:.2f}'
            )
            if context_manager:
                context_manager.info_sync(
                    f'Starting dock operation at pose '
                    f'({dock_pose.pose.position.x:.2f},\n'
                    f' {dock_pose.pose.position.y:.2f})'
                )

            self._ensure_action_server(
                self.navigator.docking_client, '/dock_robot'
            )
            self.navigator.dockRobotByPose(dock_pose, dock_type, nav_to_dock)
            dock_description = (
                f'pose ({dock_pose.pose.position.x:.2f},\n'
                f'      {dock_pose.pose.position.y:.2f})'
            )
        else:
            # Dock using ID
            self.navigator.get_logger().info(
                f'Docking robot at dock ID: {dock_id}')
            if context_manager:
                context_manager.info_sync(
                    f'Starting dock operation at dock ID: {dock_id}')

            self._ensure_action_server(
                self.navigator.docking_client, '/dock_robot'
            )
            self.navigator.dockRobotByID(dock_id, nav_to_dock)
            dock_description = f'dock ID: {dock_id}'

        self._monitor_navigation_progress(context_manager, 'dock operation')

        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            return f'Successfully docked robot at {dock_description}'
        else:
            raise create_navigation_error_from_result(result, 'Dock operation')

    def undock_robot(
        self,
        dock_type: str = '',
        context_manager: Optional[MCPContextManager] = None
    ) -> str:
        """Undock the robot from a charging station or dock.

        Parameters
        ----------
        dock_type : str, optional
            Type of dock to undock from or empty to use the default.
        context_manager : MCPContextManager
            Context manager for logging.

        Returns
        -------
        str
            Success message.

        Raises
        ------
        NavigationError
            If undocking operation fails.
        """
        self.navigator.get_logger().info('Undocking robot from dock')
        if context_manager:
            context_manager.info_sync('Starting undock operation')

        self._ensure_action_server(
            self.navigator.undocking_client, '/undock_robot'
        )
        self.navigator.undockRobot(dock_type)
        self._monitor_navigation_progress(context_manager, 'undock operation')

        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            dock_type_desc = f' (type: {dock_type})' if dock_type else ''
            return f'Successfully undocked robot{dock_type_desc}'
        else:
            raise create_navigation_error_from_result(
                result, 'Undock operation')

    def destroy(self) -> None:
        """Clean up navigator resources."""
        if self._navigator:
            self._navigator.destroy_node()
            self._navigator = None

    def _ensure_action_server(
        self, client: Any, action_name: str, timeout: float = 5.0
    ) -> None:
        """Fail fast if a BasicNavigator action server is not advertised.

        BasicNavigator's goToPose / spin / backup / driveOnHeading / etc.
        internally call ``wait_for_server()`` with NO TIMEOUT — if the
        target action server (e.g. /navigate_to_pose, /spin) is missing
        because nav2 wasn't launched fully or its lifecycle never
        activated, those calls hang indefinitely. The MCP tool then
        appears to "time out" client-side after 90-180s with no
        actionable error.

        This helper runs ``wait_for_server(timeout_sec=timeout)`` BEFORE
        the goal-sending call. If the server isn't up within the budget,
        we raise a NAV2_NOT_ACTIVE error immediately so the caller sees
        a fast, actionable failure ("nav2 not active") rather than a
        long hang.
        """
        if not client.wait_for_server(timeout_sec=timeout):
            raise NavigationError(
                f'Action server for {action_name} not available after '
                f'{timeout}s — is nav2 fully launched and lifecycle-active?',
                NavigationErrorCode.NAV2_NOT_ACTIVE,
                {'action': action_name, 'wait_timeout_sec': timeout},
            )

    def _monitor_navigation_progress(
        self, context_manager: MCPContextManager, operation_name: str
    ) -> None:
        """Monitor navigation progress and provide updates.

        Parameters
        ----------
        context_manager : MCPContextManager
            Context manager for logging.
        operation_name : str
            Name of the operation being monitored.
        """
        i = 0
        while not self.navigator.isTaskComplete():
            i += 1
            feedback = self.navigator.getFeedback()
            update_interval = self.config.navigation.feedback_update_interval
            if feedback and i % update_interval == 0:
                if hasattr(feedback, 'estimated_time_remaining'):
                    seconds = Duration.from_msg(
                        feedback.estimated_time_remaining
                    ).nanoseconds / 1e9
                    context_manager.info_sync(
                        f'{operation_name.capitalize()} - '
                        f'Estimated time of arrival: {seconds:.0f} seconds.'
                    )
                else:
                    context_manager.info_sync(
                        f'{operation_name.capitalize()} in progress...')

    def _monitor_waypoint_progress(
        self, context_manager: MCPContextManager
    ) -> None:
        """Monitor waypoint following progress.

        Parameters
        ----------
        context_manager : MCPContextManager
            Context manager for logging.
        """
        i = 0
        while not self.navigator.isTaskComplete():
            i += 1
            feedback = self.navigator.getFeedback()
            update_interval = self.config.navigation.feedback_update_interval
            if feedback and i % update_interval == 0:
                current_wp = getattr(feedback, 'current_waypoint', 'Unknown')
                context_manager.info_sync(
                    f'Currently navigating to waypoint: {current_wp}')


# Global navigation manager instance
_navigation_manager: Optional[NavigationManager] = None


def get_navigation_manager() -> NavigationManager:
    """Get or create the global navigation manager instance.

    Returns
    -------
    NavigationManager
        The global navigation manager instance.
    """
    global _navigation_manager
    if _navigation_manager is None:
        _navigation_manager = NavigationManager()
    return _navigation_manager


def get_navigator() -> BasicNavigator:
    """Get the Nav2 navigator instance.

    Returns
    -------
    BasicNavigator
        The Nav2 navigator instance.
    """
    return get_navigation_manager().navigator
