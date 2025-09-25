"""MCP server wrapping Nav2 action clients.

This module exposes tools and resources for navigation via FastMCP.
Provides comprehensive navigation capabilities including pose navigation,
waypoint following, path planning, costmap operations, and lifecycle management.
"""

import anyio
import asyncio
import json
import math
from typing import Annotated, Optional
from fastmcp import FastMCP, Context
import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.duration import Duration
from tf2_ros import Buffer, TransformListener


# Create MCP application
mcp = FastMCP('nav2-mcp-server')

# Global navigator instance for reuse
_navigator: Optional[BasicNavigator] = None

# Global TF buffer and listener for reuse
_tf_buffer: Optional[Buffer] = None
_tf_listener: Optional[TransformListener] = None


def _get_navigator() -> BasicNavigator:
    """Get or create the global navigator instance."""
    global _navigator
    if _navigator is None:
        _navigator = BasicNavigator()
    return _navigator


def _get_tf_buffer() -> Buffer:
    """Get or create the global TF buffer and listener."""
    global _tf_buffer, _tf_listener, _navigator
    if _tf_buffer is None:
        navigator = _get_navigator()
        _tf_buffer = Buffer()
        _tf_listener = TransformListener(_tf_buffer, navigator)
    return _tf_buffer


# -------------------------------
# Navigation Tools
# -------------------------------

def _navigate_to_pose_sync(
    x: float, y: float, yaw: float = 0.0, ctx: Optional[Context] = None
) -> str:
    """Navigate robot to pose synchronously in worker thread.

    Parameters
    ----------
    x : float
        X coordinate of the target pose in the `map` frame.
    y : float
        Y coordinate of the target pose in the `map` frame.
    yaw : float
        Orientation in radians.
    ctx : Context
        MCP context used for logging and progress messages (async methods).

    Returns
    -------
    str
        Human-readable status message describing the navigation result.
    """
    import math
    navigator = _get_navigator()

    # Report from thread into the async context
    if ctx:
        anyio.from_thread.run(ctx.info, 'Waiting for Nav2 to become active...')
    navigator.waitUntilNav2Active()
    if ctx:
        anyio.from_thread.run(ctx.info, 'Nav2 is active')

    goal_pose = PoseStamped()
    goal_pose.header.frame_id = 'map'
    goal_pose.header.stamp = navigator.get_clock().now().to_msg()
    goal_pose.pose.position.x = x
    goal_pose.pose.position.y = y

    # Convert yaw angle to quaternion
    goal_pose.pose.orientation.w = math.cos(yaw / 2.0)
    goal_pose.pose.orientation.z = math.sin(yaw / 2.0)

    navigator.get_logger().info(f'Navigating to pose: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}')

    navigator.goToPose(goal_pose)

    i = 0
    while not navigator.isTaskComplete():
        i += 1
        feedback = navigator.getFeedback()
        if feedback and i % 5 == 0:
            seconds = Duration.from_msg(
                feedback.estimated_time_remaining
            ).nanoseconds / 1e9
            # Forward ETA updates to the async context from this thread
            if ctx:
                anyio.from_thread.run(
                    ctx.info,
                    f'Estimated time of arrival: {seconds:.0f} seconds.'
                )

    result = navigator.getResult()
    if result == TaskResult.SUCCEEDED:
        return f'Successfully navigated to pose ({x:.2f}, {y:.2f}, {yaw:.2f})'
    elif result == TaskResult.FAILED:
        return f'Navigation failed to reach pose ({x:.2f}, {y:.2f}, {yaw:.2f})'
    elif result == TaskResult.CANCELED:
        return f'Navigation to pose ({x:.2f}, {y:.2f}, {yaw:.2f}) was canceled'
    else:
        navigator.get_logger().warning(f'Unexpected navigation result: {result}')
        return f'Navigation to pose ({x:.2f}, {y:.2f}, {yaw:.2f}) returned unknown result'


@mcp.tool(
    name='navigate_to_pose',
    description="""Navigate the robot to a specific pose (position and orientation)
    in the map frame.

    Example usage:
    - navigate to position (2.0, 3.0) with orientation 1.57
    - go to x=2 y=3 with yaw=90 degrees
    - move to coordinates (2, 3) facing north
    """,
    tags={'navigate', 'go to', 'move to', 'navigate to pose', 'position'},
    annotations={
        'title': 'Navigate To Pose',
        'readOnlyHint': False,
        'openWorldHint': False
    },
)
async def navigate_to_pose(
    x: Annotated[float, 'X coordinate of the target pose in the map frame'],
    y: Annotated[float, 'Y coordinate of the target pose in the map frame'],
    yaw: Annotated[float, 'Orientation in radians (0=east, π/2=north, π=west, 3π/2=south)'] = 0.0,
    ctx: Annotated[Optional[Context], 'MCP context used for logging and progress msgs'] = None,
) -> str:
    """Navigate robot to a specific pose with position and orientation.

    Send a NavigateToPose goal to Nav2 with the specified coordinates and orientation
    in the map frame. The robot will navigate to the target location and rotate to
    face the specified direction.

    Common usage examples:
        - navigate to position (2.0, 3.0) with orientation 1.57 radians
        - go to x=2 y=3 facing north (yaw=π/2)
        - move to coordinates (0, 0) with default orientation

    Parameters
    ----------
    x : float
        X coordinate of the target pose in the map frame.
    y : float
        Y coordinate of the target pose in the map frame.
    yaw : float, optional
        Target orientation in radians (default: 0.0 = facing east).
        Common values: 0=east, π/2=north, π=west, 3π/2=south.
    ctx : Context, optional
        MCP context used for logging and progress messages.

    Returns
    -------
    str
        Human-readable status message indicating navigation result.
        Examples: 'Successfully navigated to pose', 'Navigation failed', etc.

    Notes
    -----
    - Uses Nav2's NavigateToPose action
    - Waits for Nav2 to become active before sending goal
    - Provides periodic ETA updates via MCP context logging
    - Converts yaw angle to quaternion representation for ROS messages
    """
    return await anyio.to_thread.run_sync(
        _navigate_to_pose_sync, x, y, yaw, ctx
    )


def _follow_waypoints_sync(waypoints_str: str, ctx: Optional[Context] = None) -> str:
    """Follow a sequence of waypoints synchronously.

    Parameters
    ----------
    waypoints_str : str
        JSON string containing list of waypoint coordinates [[x1,y1], [x2,y2], ...].
    ctx : Context
        MCP context for logging.

    Returns
    -------
    str
        Status message of waypoint following result.
    """
    navigator = _get_navigator()

    if ctx:
        anyio.from_thread.run(ctx.info, 'Waiting for Nav2 to become active...')
    navigator.waitUntilNav2Active()
    if ctx:
        anyio.from_thread.run(ctx.info, 'Nav2 is active')

    try:
        waypoints_data = json.loads(waypoints_str)
        if not isinstance(waypoints_data, list):
            return 'Error: waypoints must be a list of [x, y] coordinates'

        poses = []
        for i, waypoint in enumerate(waypoints_data):
            if not isinstance(waypoint, list) or len(waypoint) != 2:
                return f'Error: waypoint {i} must be [x, y] format'

            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = navigator.get_clock().now().to_msg()
            pose.pose.position.x = float(waypoint[0])
            pose.pose.position.y = float(waypoint[1])
            pose.pose.orientation.w = 1.0
            poses.append(pose)

        navigator.get_logger().info(f'Following {len(poses)} waypoints')
        if ctx:
            anyio.from_thread.run(
                ctx.info, f'Starting waypoint following with {len(poses)} points'
            )

        navigator.followWaypoints(poses)

        i = 0
        while not navigator.isTaskComplete():
            i += 1
            feedback = navigator.getFeedback()
            if feedback and i % 5 == 0 and ctx:
                current_wp = getattr(feedback, 'current_waypoint', 'Unknown')
                anyio.from_thread.run(
                    ctx.info, f'Currently navigating to waypoint: {current_wp}'
                )

        result = navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            return f'Successfully completed waypoint following with {len(poses)} waypoints'
        elif result == TaskResult.FAILED:
            return f'Waypoint following failed after {len(poses)} waypoints'
        elif result == TaskResult.CANCELED:
            return 'Waypoint following was canceled'
        else:
            return f'Waypoint following completed with unknown result: {result}'

    except json.JSONDecodeError as e:
        return f'Error parsing waypoints JSON: {e}'
    except Exception as e:
        return f'Error during waypoint following: {e}'


@mcp.tool(
    name='follow_waypoints',
    description="""Navigate the robot through a sequence of waypoints in order.

    Example usage:
    - follow waypoints at [[0, 0], [2, 0], [2, 2], [0, 2]]
    - navigate through points (1,1), (3,1), (3,3)
    - patrol between coordinates [[-1, -1], [1, 1], [1, -1]]
    """,
    tags={'follow', 'waypoints', 'sequence', 'multiple poses', 'patrol'},
    annotations={
        'title': 'Follow Waypoints',
        'readOnlyHint': False,
        'openWorldHint': False
    },
)
async def follow_waypoints(
    waypoints: Annotated[str, 'JSON string with waypoint coordinates [[x1,y1], [x2,y2], ...]'],
    ctx: Annotated[Optional[Context], 'MCP context for logging and progress'] = None,
) -> str:
    """Follow a sequence of waypoints in the specified order.

    Navigate the robot through multiple waypoints sequentially using Nav2's
    FollowWaypoints action. The robot will visit each waypoint in order.

    Parameters
    ----------
    waypoints : str
        JSON-formatted string containing waypoint coordinates.
        Format: "[[x1, y1], [x2, y2], [x3, y3], ...]"
        Example: "[[0, 0], [2, 0], [2, 2], [0, 2]]"
    ctx : Context, optional
        MCP context for logging progress updates.

    Returns
    -------
    str
        Status message indicating the result of waypoint following.

    Common usage examples:
        - follow waypoints at [[0, 0], [2, 0], [2, 2], [0, 2]]
        - navigate through points (1,1), (3,1), (3,3)
        - patrol between coordinates [[-1, -1], [1, 1], [1, -1]]
    """
    return await anyio.to_thread.run_sync(_follow_waypoints_sync, waypoints, ctx)


def _spin_robot_sync(angle: float, ctx: Optional[Context] = None) -> str:
    """Spin robot by specified angle synchronously.

    Parameters
    ----------
    angle : float
        Angle to spin in radians (positive = counterclockwise).
    ctx : Context
        MCP context for logging.

    Returns
    -------
    str
        Status message of spin operation result.
    """
    navigator = _get_navigator()

    if ctx:
        anyio.from_thread.run(ctx.info, 'Waiting for Nav2 to become active...')
    navigator.waitUntilNav2Active()
    if ctx:
        anyio.from_thread.run(ctx.info, 'Nav2 is active')

    navigator.get_logger().info(f'Spinning robot by {angle:.2f} radians')
    if ctx:
        anyio.from_thread.run(ctx.info, f'Starting spin operation: {angle:.2f} radians')

    navigator.spin(angle)

    while not navigator.isTaskComplete():
        feedback = navigator.getFeedback()
        if feedback and ctx:
            # Spin feedback might contain progress information
            anyio.from_thread.run(ctx.info, 'Spinning in progress...')

    result = navigator.getResult()
    if result == TaskResult.SUCCEEDED:
        return f'Successfully spun robot by {angle:.2f} radians'
    elif result == TaskResult.FAILED:
        return f'Failed to spin robot by {angle:.2f} radians'
    elif result == TaskResult.CANCELED:
        return 'Spin operation was canceled'
    else:
        return f'Spin operation completed with unknown result: {result}'


@mcp.tool(
    name='spin_robot',
    description="""Rotate the robot in place by a specified angle.

    Example usage:
    - spin robot 90 degrees clockwise
    - rotate robot by π/2 radians
    - turn robot around (π radians)
    """,
    tags={'spin', 'rotate', 'turn', 'in place', 'angle'},
    annotations={
        'title': 'Spin Robot',
        'readOnlyHint': False,
        'openWorldHint': False
    },
)
async def spin_robot(
    angle: Annotated[
        float, 'Angle to spin in radians (positive=counterclockwise, negative=clockwise)'
    ],
    ctx: Annotated[Optional[Context], 'MCP context for logging'] = None,
) -> str:
    """Rotate the robot in place by the specified angle.

    Uses Nav2's Spin action to rotate the robot without changing position.
    Useful for adjusting robot orientation or performing patrol behaviors.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.
        Positive values rotate counterclockwise, negative values clockwise.
        Common values: π/2 = 90°, π = 180°, 2π = 360°.
    ctx : Context, optional
        MCP context for logging progress.

    Returns
    -------
    str
        Status message indicating the result of the spin operation.

    Examples
    --------
    - await spin_robot(1.57)  # 90 degrees counterclockwise
    'Successfully spun robot by 1.57 radians'
    - await spin_robot(-3.14)  # 180 degrees clockwise
    'Successfully spun robot by -3.14 radians'
    """
    return await anyio.to_thread.run_sync(_spin_robot_sync, angle, ctx)


def _backup_robot_sync(distance: float, speed: float, ctx: Optional[Context] = None) -> str:
    """Back up robot by specified distance synchronously.

    Parameters
    ----------
    distance : float
        Distance to back up in meters (positive values).
    speed : float
        Speed to back up in m/s.
    ctx : Context
        MCP context for logging.

    Returns
    -------
    str
        Status message of backup operation result.
    """
    navigator = _get_navigator()

    if ctx:
        anyio.from_thread.run(ctx.info, 'Waiting for Nav2 to become active...')
    navigator.waitUntilNav2Active()
    if ctx:
        anyio.from_thread.run(ctx.info, 'Nav2 is active')

    navigator.get_logger().info(f'Backing up robot: {distance:.2f}m at {speed:.2f}m/s')
    if ctx:
        anyio.from_thread.run(
            ctx.info, f'Starting backup: {distance:.2f}m at {speed:.2f}m/s'
        )

    navigator.backup(distance, speed)

    while not navigator.isTaskComplete():
        feedback = navigator.getFeedback()
        if feedback and ctx:
            anyio.from_thread.run(ctx.info, 'Backing up in progress...')

    result = navigator.getResult()
    if result == TaskResult.SUCCEEDED:
        return f'Successfully backed up {distance:.2f} meters'
    elif result == TaskResult.FAILED:
        return f'Failed to back up {distance:.2f} meters'
    elif result == TaskResult.CANCELED:
        return 'Backup operation was canceled'
    else:
        return f'Backup operation completed with unknown result: {result}'


@mcp.tool(
    name='backup_robot',
    description="""Move the robot backward by a specified distance.

    Example usage:
    - back up robot 1 meter at 0.2 m/s
    - reverse robot 0.5 meters slowly
    - backup 2 meters at normal speed
    """,
    tags={'backup', 'reverse', 'back up', 'move backward'},
    annotations={
        'title': 'Backup Robot',
        'readOnlyHint': False,
        'openWorldHint': False
    },
)
async def backup_robot(
    distance: Annotated[float, 'Distance to back up in meters (positive value)'],
    speed: Annotated[float, 'Backup speed in m/s (typically 0.1-0.5)'] = 0.2,
    ctx: Annotated[Optional[Context], 'MCP context for logging'] = None,
) -> str:
    """Move the robot backward in a straight line.

    Uses Nav2's Backup action to move the robot backward by the specified
    distance at the given speed. Useful for getting unstuck or repositioning.

    Parameters
    ----------
    distance : float
        Distance to back up in meters (must be positive).
    speed : float, optional
        Backup speed in m/s (default: 0.2 m/s).
        Typical range: 0.1-0.5 m/s for safety.
    ctx : Context, optional
        MCP context for logging progress.

    Returns
    -------
    str
        Status message indicating the result of the backup operation.
    """
    return await anyio.to_thread.run_sync(_backup_robot_sync, distance, speed, ctx)


# -------------------------------
# Costmap Management Tools
# -------------------------------

def _clear_costmaps_sync(costmap_type: str, ctx: Optional[Context] = None) -> str:
    """Clear costmaps synchronously.

    Parameters
    ----------
    costmap_type : str
        Type of costmap to clear: 'global', 'local', or 'all'.
    ctx : Context
        MCP context for logging.

    Returns
    -------
    str
        Status message of clear operation result.
    """
    navigator = _get_navigator()

    if ctx:
        anyio.from_thread.run(ctx.info, f'Clearing {costmap_type} costmap(s)...')

    try:
        if costmap_type == 'global':
            navigator.clearGlobalCostmap()
            result_msg = 'Global costmap cleared successfully'
        elif costmap_type == 'local':
            navigator.clearLocalCostmap()
            result_msg = 'Local costmap cleared successfully'
        elif costmap_type == 'all':
            navigator.clearAllCostmaps()
            result_msg = 'All costmaps cleared successfully'
        else:
            return f'Error: Invalid costmap type "{costmap_type}". Use: global, local, or all'

        if ctx:
            anyio.from_thread.run(ctx.info, result_msg)
        return result_msg

    except Exception as e:
        error_msg = f'Failed to clear {costmap_type} costmap(s): {e}'
        if ctx:
            anyio.from_thread.run(ctx.error, error_msg)
        return error_msg


@mcp.tool(
    name='clear_costmaps',
    description="""Clear robot navigation costmaps to remove stale obstacle data.

    Example usage:
    - clear all costmaps
    - clear global costmap only
    - reset local costmap
    """,
    tags={'clear', 'costmap', 'obstacles', 'reset', 'navigation'},
    annotations={
        'title': 'Clear Costmaps',
        'readOnlyHint': False,
        'openWorldHint': False
    },
)
async def clear_costmaps(
    costmap_type: Annotated[str, 'Type of costmap to clear: "global", "local", or "all"'] = 'all',
    ctx: Annotated[Optional[Context], 'MCP context for logging'] = None,
) -> str:
    """Clear navigation costmaps to remove stale obstacle information.

    Clears the specified costmap(s) to help resolve navigation issues caused
    by outdated obstacle data. Useful when the robot gets stuck or when
    obstacles have moved since last observation.

    Parameters
    ----------
    costmap_type : str, optional
        Which costmap to clear (default: 'all'):
        - 'global': Clear only the global costmap
        - 'local': Clear only the local costmap
        - 'all': Clear both global and local costmaps
    ctx : Context, optional
        MCP context for logging progress.

    Returns
    -------
    str
        Status message indicating the result of the clear operation.

    Notes
    -----
    - Global costmap: Used for long-term path planning
    - Local costmap: Used for immediate obstacle avoidance
    - Clearing costmaps removes dynamic obstacles but keeps static map data
    """
    return await anyio.to_thread.run_sync(_clear_costmaps_sync, costmap_type, ctx)


# -------------------------------
# Status and Information Tools
# -------------------------------

def _get_robot_pose_sync(ctx: Optional[Context] = None) -> str:
    """Get current robot pose synchronously.

    Parameters
    ----------
    ctx : Context
        MCP context for logging.

    Returns
    -------
    str
        JSON string with current robot pose information.
    """
    try:
        if ctx:
            anyio.from_thread.run(ctx.info, 'Getting current robot pose...')

        # Get global TF buffer
        tf_buffer = _get_tf_buffer()

        try:
            # Get transform from base_link to map
            transform = tf_buffer.lookup_transform(
                'map',  # target frame
                'base_link',  # source frame
                rclpy.time.Time(),  # get the latest available transform
                timeout=rclpy.duration.Duration(seconds=2.0)
            )

            # Extract position and orientation from transform
            quat_w = transform.transform.rotation.w
            quat_x = transform.transform.rotation.x
            quat_y = transform.transform.rotation.y
            quat_z = transform.transform.rotation.z

            # Calculate yaw from quaternion (rotation around z-axis)
            yaw = math.atan2(2.0 * (quat_w * quat_z + quat_x * quat_y),
                             1.0 - 2.0 * (quat_y * quat_y + quat_z * quat_z))

            pose_info = {
                'x': transform.transform.translation.x,
                'y': transform.transform.translation.y,
                'yaw': yaw,
            }

        except Exception as tf_error:
            # If transform is not available, return error information
            pose_info = {
                'status': 'transform_unavailable',
                'message': f'Could not get transform from base_link to map: {str(tf_error)}',
                'error_type': type(tf_error).__name__
            }

        result = json.dumps(pose_info, indent=2)
        if ctx:
            anyio.from_thread.run(ctx.info, f'Current robot pose: {result}')
        return result

    except Exception as e:
        error_msg = f'Error getting robot pose: {e}'
        if ctx:
            anyio.from_thread.run(ctx.error, error_msg)
        return json.dumps({'status': 'error', 'message': error_msg})


@mcp.tool(
    name='get_robot_pose',
    description="""Get the current position and orientation of the robot.

    Example usage:
    - where is the robot now?
    - get current robot position
    - show robot pose
    """,
    tags={'pose', 'position', 'location', 'robot', 'current', 'where'},
    annotations={
        'title': 'Get Robot Pose',
        'readOnlyHint': True,
        'openWorldHint': False
    },
)
async def get_robot_pose(
    ctx: Annotated[Optional[Context], 'MCP context for logging'] = None,
) -> str:
    """Get the current pose (position and orientation) of the robot.

    Retrieves the robot's current position and orientation in the map frame.
    Returns detailed pose information including coordinates and quaternion orientation.

    Parameters
    ----------
    ctx : Context, optional
        MCP context for logging.

    Returns
    -------
    str
        JSON string containing current robot pose information with fields:
        - x, y, z: Position coordinates in map frame
        - orientation_w, orientation_x, orientation_y, orientation_z: Quaternion
        - frame_id: Reference frame (typically 'map')
        - status: Availability status of pose data
    """
    return await anyio.to_thread.run_sync(_get_robot_pose_sync, ctx)


def _get_navigation_status_sync(ctx: Optional[Context] = None) -> str:
    """Get current navigation status synchronously.

    Parameters
    ----------
    ctx : Context
        MCP context for logging.

    Returns
    -------
    str
        JSON string with navigation status information.
    """
    navigator = _get_navigator()

    try:
        if ctx:
            anyio.from_thread.run(ctx.info, 'Getting navigation status...')

        # Check if Nav2 is active
        nav2_active = False
        try:
            navigator.waitUntilNav2Active(localisation_timeout_sec=1.0)
            nav2_active = True
        except Exception:
            nav2_active = False

        # Get task status
        is_task_complete = navigator.isTaskComplete()

        # Get feedback if available
        feedback_info = None
        if not is_task_complete:
            feedback = navigator.getFeedback()
            if feedback:
                feedback_info = {
                    'has_feedback': True,
                    'feedback_type': str(type(feedback).__name__)
                }
                if hasattr(feedback, 'distance_remaining'):
                    feedback_info['distance_remaining'] = feedback.distance_remaining
                if hasattr(feedback, 'estimated_time_remaining'):
                    eta_duration = Duration.from_msg(feedback.estimated_time_remaining)
                    feedback_info['estimated_time_remaining_seconds'] = (
                        eta_duration.nanoseconds / 1e9
                    )

        # Get error information
        error_code, error_msg = navigator.getTaskError()

        status_info = {
            'nav2_active': nav2_active,
            'task_complete': is_task_complete,
            'error_code': error_code,
            'error_message': error_msg if error_msg else None,
            'feedback': feedback_info,
            'timestamp': navigator.get_clock().now().to_msg()
        }

        result = json.dumps(status_info, indent=2, default=str)
        if ctx:
            anyio.from_thread.run(ctx.info, f'Navigation status: {result}')
        return result

    except Exception as e:
        error_msg = f'Error getting navigation status: {e}'
        if ctx:
            anyio.from_thread.run(ctx.error, error_msg)
        return json.dumps({'status': 'error', 'message': error_msg})


@mcp.tool(
    name='get_navigation_status',
    description="""Get the current status of the navigation system and any active tasks.

    Example usage:
    - what is the navigation status?
    - is the robot currently navigating?
    - show navigation progress
    """,
    tags={'status', 'navigation', 'nav2', 'active', 'task', 'progress'},
    annotations={
        'title': 'Get Navigation Status',
        'readOnlyHint': True,
        'openWorldHint': False
    },
)
async def get_navigation_status(
    ctx: Annotated[Optional[Context], 'MCP context for logging'] = None,
) -> str:
    """Get comprehensive information about the current navigation status.

    Returns detailed status information including Nav2 activation state,
    task completion status, error information, and current feedback data.

    Parameters
    ----------
    ctx : Context, optional
        MCP context for logging.

    Returns
    -------
    str
        JSON string containing navigation status with fields:
        - nav2_active: Whether Nav2 is currently active
        - task_complete: Whether current navigation task is complete
        - error_code: Error code from last navigation task
        - error_message: Human-readable error message
        - feedback: Current navigation feedback (if available)
        - timestamp: Current system timestamp
    """
    return await anyio.to_thread.run_sync(_get_navigation_status_sync, ctx)


def _cancel_navigation_sync(ctx: Optional[Context] = None) -> str:
    """Cancel current navigation task synchronously.

    Parameters
    ----------
    ctx : Context
        MCP context for logging.

    Returns
    -------
    str
        Status message of cancellation operation.
    """
    navigator = _get_navigator()

    try:
        if ctx:
            anyio.from_thread.run(ctx.info, 'Canceling current navigation task...')

        # Check if there's an active task
        if navigator.isTaskComplete():
            result_msg = 'No active navigation task to cancel'
        else:
            navigator.cancelTask()
            result_msg = 'Navigation task cancellation requested'

        if ctx:
            anyio.from_thread.run(ctx.info, result_msg)
        return result_msg

    except Exception as e:
        error_msg = f'Error canceling navigation: {e}'
        if ctx:
            anyio.from_thread.run(ctx.error, error_msg)
        return error_msg


@mcp.tool(
    name='cancel_navigation',
    description="""Cancel the currently active navigation task.

    Example usage:
    - cancel current navigation
    - stop the robot navigation
    - abort navigation task
    """,
    tags={'cancel', 'stop', 'abort', 'navigation', 'task'},
    annotations={
        'title': 'Cancel Navigation',
        'readOnlyHint': False,
        'openWorldHint': False
    },
)
async def cancel_navigation(
    ctx: Annotated[Optional[Context], 'MCP context for logging'] = None,
) -> str:
    """Cancel any currently active navigation task.

    Immediately stops the robot's current navigation activity.
    Safe to call even if no navigation task is active.

    Parameters
    ----------
    ctx : Context, optional
        MCP context for logging.

    Returns
    -------
    str
        Status message indicating the result of the cancellation request.
    """
    return await anyio.to_thread.run_sync(_cancel_navigation_sync, ctx)


# -------------------------------
# Lifecycle Management Tools
# -------------------------------

def _lifecycle_operation_sync(operation: str, ctx: Optional[Context] = None) -> str:
    """Perform lifecycle operation synchronously.

    Parameters
    ----------
    operation : str
        Lifecycle operation: 'startup' or 'shutdown'.
    ctx : Context
        MCP context for logging.

    Returns
    -------
    str
        Status message of lifecycle operation result.
    """
    navigator = _get_navigator()

    try:
        if ctx:
            anyio.from_thread.run(ctx.info, f'Performing Nav2 {operation}...')

        if operation == 'startup':
            navigator.lifecycleStartup()
            result_msg = 'Nav2 lifecycle startup completed successfully'
        elif operation == 'shutdown':
            navigator.lifecycleShutdown()
            result_msg = 'Nav2 lifecycle shutdown completed successfully'
        else:
            return f'Error: Invalid operation "{operation}". Use: startup or shutdown'

        if ctx:
            anyio.from_thread.run(ctx.info, result_msg)
        return result_msg

    except Exception as e:
        error_msg = f'Failed to {operation} Nav2 lifecycle: {e}'
        if ctx:
            anyio.from_thread.run(ctx.error, error_msg)
        return error_msg


@mcp.tool(
    name='nav2_lifecycle',
    description="""Control Nav2 lifecycle (startup or shutdown).

    Example usage:
    - startup nav2 system
    - shutdown navigation
    - activate nav2 lifecycle
    """,
    tags={'lifecycle', 'startup', 'shutdown', 'nav2', 'activate', 'deactivate'},
    annotations={
        'title': 'Nav2 Lifecycle Control',
        'readOnlyHint': False,
        'openWorldHint': False
    },
)
async def nav2_lifecycle(
    operation: Annotated[str, 'Lifecycle operation: "startup" or "shutdown"'],
    ctx: Annotated[Optional[Context], 'MCP context for logging'] = None,
) -> str:
    """Control the Nav2 navigation system lifecycle.

    Manage the startup and shutdown of Nav2 navigation components.
    Useful for initializing or cleanly shutting down the navigation system.

    Parameters
    ----------
    operation : str
        The lifecycle operation to perform:
        - 'startup': Initialize and activate Nav2 components
        - 'shutdown': Deactivate and shutdown Nav2 components
    ctx : Context, optional
        MCP context for logging progress.

    Returns
    -------
    str
        Status message indicating the result of the lifecycle operation.

    Notes
    -----
    - Startup operation waits for all Nav2 nodes to become active
    - Shutdown operation cleanly deactivates all Nav2 lifecycle nodes
    - These operations may take several seconds to complete
    """
    return await anyio.to_thread.run_sync(_lifecycle_operation_sync, operation, ctx)


# -------------------------------
# Logging Configuration
# -------------------------------

def setup_logging():
    """Configure logging for the MCP server."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


# -------------------------------
# Server Configuration
# -------------------------------

@mcp.resource(
    uri='nav2://status',
    name='Navigation Status',
    description='Current status of the navigation system',
    mime_type='application/json'
)
async def get_nav_status_resource() -> str:
    """Resource endpoint for navigation status."""
    return _get_navigation_status_sync()


@mcp.resource(
    uri='nav2://pose',
    name='Robot Pose',
    description='Current robot pose in map frame',
    mime_type='application/json'
)
async def get_robot_pose_resource() -> str:
    """Resource endpoint for robot pose."""
    return _get_robot_pose_sync()


async def main() -> None:
    """Run the Nav2 MCP server.

    Initializes ROS2, sets up logging, and starts the MCP server
    with stdio transport for integration with MCP clients.

    Notes
    -----
    - Uses stdio transport for local MCP integration
    - Configures structured logging for debugging
    - Initializes global navigator instance
    - Handles graceful shutdown of ROS2 nodes
    """
    # Setup logging
    logger = setup_logging()
    logger.info('Starting Nav2 MCP Server...')

    # Initialize ROS2
    rclpy.init()
    logger.info('ROS2 initialized')

    try:
        # Pre-initialize navigator to check ROS2 connectivity
        _get_navigator()
        logger.info('Navigator initialized')

        # Start MCP server
        logger.info('Starting MCP server on stdio transport')
        await mcp.run_async(transport='stdio')

    except KeyboardInterrupt:
        logger.info('Server interrupted by user')
    except Exception as e:
        logger.error(f'Server error: {e}')
        raise
    finally:
        # Clean up ROS2
        logger.info('Shutting down ROS2...')
        global _tf_listener, _tf_buffer
        if _tf_listener:
            _tf_listener = None
        if _tf_buffer:
            _tf_buffer = None
        if _navigator:
            _navigator.destroy_node()
        rclpy.shutdown()
        logger.info('Nav2 MCP Server shutdown complete')


if __name__ == '__main__':
    asyncio.run(main())
