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

"""MCP tools for Nav2 navigation operations.

This module provides the MCP tool definitions that wrap the navigation
and transform operations with proper async/sync handling.
"""

from typing import Annotated, Optional
import anyio
from fastmcp import FastMCP, Context

from .config import get_config
from .navigation import get_navigation_manager
from .transforms import get_transform_manager
from .utils import (
    MCPContextManager,
    with_context_logging,
    with_nav2_active_check,
    safe_json_dumps
)


def create_mcp_tools(mcp: FastMCP) -> None:
    """Create and register all MCP tools with the FastMCP instance.

    Parameters
    ----------
    mcp : FastMCP
        The FastMCP server instance to register tools with.
    """
    config = get_config()

    # -------------------------------
    # Navigation Tools
    # -------------------------------

    @mcp.tool(
        name='navigate_to_pose',
        description="""Navigate the robot to a specific pose (position and
        orientation) in the map frame.

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
        x: Annotated[float, 'X coordinate of target pose in map frame'],
        y: Annotated[float, 'Y coordinate of target pose in map frame'],
        yaw: Annotated[
            float,
            'Orientation in radians (0=east, π/2=north, π=west, 3π/2=south)'
        ] = 0.0,
        ctx: Annotated[
            Optional[Context],
            'MCP context used for logging and progress msgs'
        ] = None,
    ) -> str:
        """Navigate robot to a specific pose with position and orientation."""
        return await anyio.to_thread.run_sync(
            _navigate_to_pose_sync, x, y, yaw, ctx
        )

    @mcp.tool(
        name='follow_waypoints',
        description="""Navigate the robot through a sequence of waypoints
        in order.

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
        waypoints: Annotated[
            str,
            'JSON string with waypoint coordinates [[x1,y1], [x2,y2], ...]'
        ],
        ctx: Annotated[
            Optional[Context],
            'MCP context for logging and progress'
        ] = None,
    ) -> str:
        """Follow a sequence of waypoints in the specified order."""
        return await anyio.to_thread.run_sync(
            _follow_waypoints_sync, waypoints, ctx
        )

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
            float,
            'Angle to spin in radians (positive=counterclockwise, '
            'negative=clockwise)'
        ],
        ctx: Annotated[Optional[Context], 'MCP context for logging'] = None,
    ) -> str:
        """Rotate the robot in place by the specified angle."""
        return await anyio.to_thread.run_sync(_spin_robot_sync, angle, ctx)

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
        distance: Annotated[
            float, 'Distance to back up in meters (positive value)'
        ],
        speed: Annotated[
            float,
            'Backup speed in m/s (typically 0.1-0.5)'
        ] = None,
        ctx: Annotated[Optional[Context], 'MCP context for logging'] = None,
    ) -> str:
        """Move the robot backward in a straight line."""
        if speed is None:
            speed = config.navigation.default_backup_speed
        return await anyio.to_thread.run_sync(
            _backup_robot_sync, distance, speed, ctx
        )

    @mcp.tool(
        name='dock_robot',
        description="""Dock the robot to a charging station or dock.

        This tool allows docking the robot either by specifying exact
        coordinates or by using a predefined dock ID. The robot can
        optionally navigate to a staging pose before performing the
        docking maneuver.

        Example usage:
        - dock robot at coordinates (5.0, 2.0) with 0 degree orientation
        - dock robot at dock ID "dock_01"
        - dock robot with specific dock type or empty to use the default
        """,
        tags={'dock', 'docking', 'charging', 'station', 'park'},
        annotations={
            'title': 'Dock Robot',
            'readOnlyHint': False,
            'openWorldHint': False
        },
    )
    async def dock_robot(
        x: Annotated[float, 'X coordinate of dock in map frame'] = None,
        y: Annotated[float, 'Y coordinate of dock in map frame'] = None,
        yaw: Annotated[
            float, 'Orientation at dock in radians (0 = facing east)'
        ] = 0.0,
        dock_id: Annotated[
            str, 'ID of predefined dock (alternative to coordinates)'
        ] = '',
        dock_type: Annotated[
            str, 'Type of dock or empty to use the default'
        ] = '',
        nav_to_dock: Annotated[
            bool, 'Whether to navigate to staging pose before docking'
        ] = True,
        ctx: Annotated[Optional[Context], 'MCP context for logging'] = None,
    ) -> str:
        """Dock the robot to a charging station or dock."""
        return await anyio.to_thread.run_sync(
            _dock_robot_sync, x, y, yaw, dock_id, dock_type, nav_to_dock, ctx
        )

    @mcp.tool(
        name='undock_robot',
        description="""Undock the robot from a charging station or dock.

        This tool allows the robot to undock from its current docking position
        and move to a safe undocked state.

        Example usage:
        - undock robot from current dock
        - undock robot from specific dock type
        """,
        tags={'undock', 'undocking', 'leave dock', 'depart'},
        annotations={
            'title': 'Undock Robot',
            'readOnlyHint': False,
            'openWorldHint': False
        },
    )
    async def undock_robot(
        dock_type: Annotated[
            str, 'Type of dock to undock from or empty to use default'
        ] = '',
        ctx: Annotated[Optional[Context], 'MCP context for logging'] = None,
    ) -> str:
        """Undock the robot from a charging station or dock."""
        return await anyio.to_thread.run_sync(
            _undock_robot_sync, dock_type, ctx
        )

    @mcp.tool(
        name='get_path',
        description="""Compute a navigation path between two poses
        (start and goal) in the map frame, using the planner.

        Example usage:
        - get path from (x1, y1, yaw1) to (x2, y2, yaw2)
        - plan path between two positions
        - compute path for robot navigation
        """,
        tags={'path', 'plan', 'compute', 'navigation', 'route', 'planner'},
        annotations={
            'title': 'Get Path',
            'readOnlyHint': True,
            'openWorldHint': False
        },
    )
    async def get_path(
        start_x: Annotated[float, 'X coordinate of start pose in map frame'],
        start_y: Annotated[float, 'Y coordinate of start pose in map frame'],
        goal_x: Annotated[float, 'X coordinate of goal pose in map frame'],
        goal_y: Annotated[float, 'Y coordinate of goal pose in map frame'],
        start_yaw: Annotated[
            float, 'Orientation of start pose in radians'] = 0.0,
        goal_yaw: Annotated[
            float, 'Orientation of goal pose in radians'] = 0.0,
        planner_id: Annotated[str, 'Planner ID to use (optional)'] = '',
        ctx: Annotated[Optional[Context], 'MCP context for logging'] = None,
    ) -> str:
        """Compute a navigation path between two poses."""
        return await anyio.to_thread.run_sync(
            _get_path_sync,
            start_x, start_y, start_yaw,
            goal_x, goal_y, goal_yaw,
            planner_id, True, ctx
        )

    @mcp.tool(
        name='get_path_from_robot',
        description="""Compute a navigation path between the robot's
        current pose and a goal pose in the map frame, using the planner.

        Example usage:
        - get path from robot to (x2, y2, yaw2)
        - plan path to (x2, y2, yaw2)
        - compute path for robot navigation
        """,
        tags={'path', 'plan', 'compute', 'navigation', 'route', 'planner'},
        annotations={
            'title': 'Get Path',
            'readOnlyHint': True,
            'openWorldHint': False
        },
    )
    async def get_path_from_robot(
        goal_x: Annotated[float, 'X coordinate of goal pose in map frame'],
        goal_y: Annotated[float, 'Y coordinate of goal pose in map frame'],
        goal_yaw: Annotated[
            float, 'Orientation of goal pose in radians'] = 0.0,
        planner_id: Annotated[str, 'Planner ID to use (optional)'] = '',
        ctx: Annotated[Optional[Context], 'MCP context for logging'] = None,
    ) -> str:
        """Compute a navigation path between two poses."""
        return await anyio.to_thread.run_sync(
            _get_path_sync,
            goal_x, goal_y, goal_yaw,
            goal_x, goal_y, goal_yaw,
            planner_id, False, ctx
        )

    # -------------------------------
    # Costmap Management Tools
    # -------------------------------

    @mcp.tool(
        name='clear_costmaps',
        description="""Clear robot navigation costmaps to remove stale
        obstacle data.

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
        costmap_type: Annotated[
            str,
            'Type of costmap to clear: "global", "local", or "all"'
        ] = 'all',
        ctx: Annotated[Optional[Context], 'MCP context for logging'] = None,
    ) -> str:
        """Clear navigation costmaps to remove stale obstacle information."""
        return await anyio.to_thread.run_sync(
            _clear_costmaps_sync, costmap_type, ctx
        )

    # -------------------------------
    # Status and Information Tools
    # -------------------------------

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
        """Get the current pose (position and orientation) of the robot."""
        return await anyio.to_thread.run_sync(_get_robot_pose_sync, ctx)

    @mcp.tool(
        name='get_navigation_status',
        description="""Get the current status of the navigation system
        and any active tasks.

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
        """Get comprehensive information about current navigation status."""
        return await anyio.to_thread.run_sync(_get_navigation_status_sync, ctx)

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
        """Cancel any currently active navigation task."""
        return await anyio.to_thread.run_sync(_cancel_navigation_sync, ctx)

    # -------------------------------
    # Lifecycle Management Tools
    # -------------------------------

    @mcp.tool(
        name='nav2_lifecycle',
        description="""Control Nav2 lifecycle (startup or shutdown).

        Example usage:
        - startup nav2 system
        - shutdown navigation
        - activate nav2 lifecycle
        """,
        tags={'lifecycle', 'startup', 'shutdown',
              'nav2', 'activate', 'deactivate'},
        annotations={
            'title': 'Nav2 Lifecycle Control',
            'readOnlyHint': False,
            'openWorldHint': False
        },
    )
    async def nav2_lifecycle(
        operation: Annotated[
            str, 'Lifecycle operation: "startup" or "shutdown"'
        ],
        ctx: Annotated[Optional[Context], 'MCP context for logging'] = None,
    ) -> str:
        """Control the Nav2 navigation system lifecycle."""
        return await anyio.to_thread.run_sync(
            _nav2_lifecycle_sync, operation, ctx
        )


# -------------------------------
# Synchronous Implementation Functions
# -------------------------------

@with_context_logging
@with_nav2_active_check
def _navigate_to_pose_sync(
    x: float, y: float, yaw: float = 0.0,
    ctx: Optional[Context] = None,
    context_manager: Optional[MCPContextManager] = None
) -> str:
    """Navigate robot to pose synchronously."""
    nav_manager = get_navigation_manager()
    return nav_manager.navigate_to_pose(x, y, yaw, context_manager)


@with_context_logging
@with_nav2_active_check
def _follow_waypoints_sync(
    waypoints_str: str,
    ctx: Optional[Context] = None,
    context_manager: Optional[MCPContextManager] = None
) -> str:
    """Follow waypoints synchronously."""
    nav_manager = get_navigation_manager()
    return nav_manager.follow_waypoints(waypoints_str, context_manager)


@with_context_logging
@with_nav2_active_check
def _spin_robot_sync(
    angle: float,
    ctx: Optional[Context] = None,
    context_manager: Optional[MCPContextManager] = None
) -> str:
    """Spin robot synchronously."""
    nav_manager = get_navigation_manager()
    return nav_manager.spin_robot(angle, context_manager)


@with_context_logging
@with_nav2_active_check
def _backup_robot_sync(
    distance: float, speed: float,
    ctx: Optional[Context] = None,
    context_manager: Optional[MCPContextManager] = None
) -> str:
    """Back up robot synchronously."""
    nav_manager = get_navigation_manager()
    return nav_manager.backup_robot(distance, speed, context_manager)


@with_context_logging
def _clear_costmaps_sync(
    costmap_type: str,
    ctx: Optional[Context] = None,
    context_manager: Optional[MCPContextManager] = None
) -> str:
    """Clear costmaps synchronously."""
    nav_manager = get_navigation_manager()
    return nav_manager.clear_costmaps(costmap_type, context_manager)


@with_context_logging
def _get_robot_pose_sync(
    ctx: Optional[Context] = None,
    context_manager: Optional[MCPContextManager] = None
) -> str:
    """Get robot pose synchronously."""
    transform_manager = get_transform_manager()
    pose_info = transform_manager.get_robot_pose(context_manager)
    return safe_json_dumps(pose_info)


@with_context_logging
def _get_navigation_status_sync(
    ctx: Optional[Context] = None,
    context_manager: Optional[MCPContextManager] = None
) -> str:
    """Get navigation status synchronously."""
    nav_manager = get_navigation_manager()
    status_info = nav_manager.get_navigation_status(context_manager)
    return safe_json_dumps(status_info)


@with_context_logging
def _cancel_navigation_sync(
    ctx: Optional[Context] = None,
    context_manager: Optional[MCPContextManager] = None
) -> str:
    """Cancel navigation synchronously."""
    nav_manager = get_navigation_manager()
    return nav_manager.cancel_navigation(context_manager)


@with_context_logging
def _nav2_lifecycle_sync(
    operation: str,
    ctx: Optional[Context] = None,
    context_manager: Optional[MCPContextManager] = None
) -> str:
    """Perform Nav2 lifecycle operation synchronously."""
    nav_manager = get_navigation_manager()

    if operation == 'startup':
        return nav_manager.lifecycle_startup(context_manager)
    elif operation == 'shutdown':
        return nav_manager.lifecycle_shutdown(context_manager)
    else:
        raise ValueError(
            f'Invalid operation "{operation}". Use: startup or shutdown')


@with_context_logging
def _dock_robot_sync(
    x: Optional[float],
    y: Optional[float],
    yaw: float,
    dock_id: str,
    dock_type: str,
    nav_to_dock: bool,
    ctx: Optional[Context] = None,
    context_manager: Optional[MCPContextManager] = None
) -> str:
    """Dock robot synchronously."""
    nav_manager = get_navigation_manager()

    # Create dock pose if coordinates are provided
    dock_pose = None
    if x is not None and y is not None:
        dock_pose = nav_manager.create_pose_stamped(x, y, yaw)

    return nav_manager.dock_robot(
        dock_pose=dock_pose,
        dock_id=dock_id,
        dock_type=dock_type,
        nav_to_dock=nav_to_dock,
        context_manager=context_manager
    )


@with_context_logging
def _undock_robot_sync(
    dock_type: str,
    ctx: Optional[Context] = None,
    context_manager: Optional[MCPContextManager] = None
) -> str:
    """Undock robot synchronously."""
    nav_manager = get_navigation_manager()
    return nav_manager.undock_robot(dock_type, context_manager)


@with_context_logging
def _get_path_sync(
    start_x: float,
    start_y: float,
    start_yaw: float,
    goal_x: float,
    goal_y: float,
    goal_yaw: float,
    planner_id: str,
    use_start: bool,
    ctx: Optional[Context] = None,
    context_manager: Optional[MCPContextManager] = None
) -> str:
    """Get path synchronously."""
    nav_manager = get_navigation_manager()
    return nav_manager.get_path(
        start_x, start_y, start_yaw, goal_x, goal_y, goal_yaw, planner_id,
        use_start, context_manager
    )
