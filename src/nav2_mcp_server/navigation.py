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
from typing import List, Optional, Dict, Any
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration

from .config import get_config
from .exceptions import (
    NavigationError,
    NavigationErrorCode,
    create_navigation_error_from_result
)
from .utils import MCPContextManager, validate_numeric_range


class NavigationManager:
    """Manages Nav2 navigation operations and state."""

    def __init__(self):
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
                f"Invalid waypoints JSON format: {e}",
                NavigationErrorCode.INVALID_WAYPOINTS,
                {'json_error': str(e)}
            )

        if not isinstance(waypoints_data, list):
            raise NavigationError(
                "Waypoints must be a list of [x, y] coordinates",
                NavigationErrorCode.INVALID_WAYPOINTS,
                {'received_type': type(waypoints_data).__name__}
            )

        if len(waypoints_data) > self.config.navigation.max_waypoints:
            raise NavigationError(
                f"Too many waypoints. Maximum: "
                f"{self.config.navigation.max_waypoints}",
                NavigationErrorCode.INVALID_WAYPOINTS,
                {'waypoint_count': len(waypoints_data)}
            )

        poses = []
        for i, waypoint in enumerate(waypoints_data):
            if not isinstance(waypoint, list) or len(waypoint) != 2:
                raise NavigationError(
                    f"Waypoint {i} must be [x, y] format",
                    NavigationErrorCode.INVALID_WAYPOINTS,
                    {'waypoint_index': i, 'waypoint_data': waypoint}
                )

            try:
                x, y = float(waypoint[0]), float(waypoint[1])
                poses.append(self.create_pose_stamped(x, y))
            except (ValueError, TypeError) as e:
                raise NavigationError(
                    f"Invalid coordinates in waypoint {i}: {e}",
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

        self.navigator.backup(distance, speed)
        self._monitor_navigation_progress(context_manager, 'backup operation')

        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            return f'Successfully backed up {distance:.2f} meters'
        else:
            raise create_navigation_error_from_result(
                result, 'Backup operation')

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
                f"Failed to clear {costmap_type} costmap(s): {str(e)}",
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
                f"Failed to startup Nav2 lifecycle: {str(e)}",
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
                f"Failed to shutdown Nav2 lifecycle: {str(e)}",
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
        context_manager: MCPContextManager = None
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
            raise ValueError("Either dock_pose or dock_id must be provided")

        if dock_pose is not None:
            # Dock using pose
            self.navigator.get_logger().info(
                f'Docking robot at pose: x={dock_pose.pose.position.x:.2f}, '
                f'y={dock_pose.pose.position.y:.2f}'
            )
            context_manager.info_sync(
                f'Starting dock operation at pose '
                f'({dock_pose.pose.position.x:.2f},\n'
                f' {dock_pose.pose.position.y:.2f})'
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
            context_manager.info_sync(
                f'Starting dock operation at dock ID: {dock_id}')

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
        context_manager: MCPContextManager = None
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
        context_manager.info_sync('Starting undock operation')

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
