"""MCP server wrapping Nav2 action clients.

This module exposes tools and resources for navigation via FastMCP.
"""

import asyncio
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose, FollowWaypoints
from nav_msgs.msg import Odometry

from fastmcp import FastMCP


# Create MCP application
mcp = FastMCP("nav2-mcp-server")


class Nav2Client(Node):
    """Helper ROS2 node that holds action clients and last odometry pose."""

    def __init__(self) -> None:
        super().__init__("nav2_mcp_client")
        self.nav_to_pose_client: ActionClient = ActionClient(
            self, NavigateToPose, "navigate_to_pose"
        )
        self.waypoints_client: ActionClient = ActionClient(
            self, FollowWaypoints, "follow_waypoints"
        )

        # Last known robot pose (from odometry)
        self.pose: Optional[Any] = None
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10
        )

    def odom_callback(self, msg: Odometry) -> None:
        """Callback that stores the last odometry pose."""
        self.pose = msg.pose.pose


rclpy.init()
nav_node = Nav2Client()


# -------------------------------
# Tools (actions)
# -------------------------------


@mcp.tool(
    name="go_to",
    description="Send a navigation goal to Nav2",
    annotations={
        "title": "Go To",
        "readOnlyHint": True,
        "openWorldHint": False,
    },
)
async def go_to(x: float, y: float, yaw: float = 0.0) -> Dict[str, Any]:
    """Send a NavigateToPose goal to Nav2.

    Note: orientation is simplified (no real yaw conversion) for brevity.
    """
    goal_msg = NavigateToPose.Goal()
    pose = PoseStamped()
    pose.header.frame_id = "map"
    pose.header.stamp = nav_node.get_clock().now().to_msg()
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.orientation.w = 1.0  # simplified (no real yaw conversion)
    goal_msg.pose = pose

    nav_node.nav_to_pose_client.wait_for_server()
    future = nav_node.nav_to_pose_client.send_goal_async(goal_msg)
    result = await asyncio.wrap_future(future)
    return {"accepted": getattr(result, "accepted", False)}


@mcp.tool(
    name="set_waypoints",
    description="Send a list of waypoints to Nav2",
    annotations={
        "title": "Set Waypoints",
        "readOnlyHint": False,
        "openWorldHint": False,
    },
)
async def set_waypoints(waypoints: List[Dict[str, float]]) -> Dict[str, Any]:
    """Send a FollowWaypoints goal containing the provided waypoints.

    Each waypoint must be a dict with numeric keys 'x' and 'y'.
    """
    goal_msg = FollowWaypoints.Goal()
    for wp in waypoints:
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.header.stamp = nav_node.get_clock().now().to_msg()
        ps.pose.position.x = float(wp.get("x", 0.0))
        ps.pose.position.y = float(wp.get("y", 0.0))
        ps.pose.orientation.w = 1.0
        goal_msg.poses.append(ps)

    nav_node.waypoints_client.wait_for_server()
    future = nav_node.waypoints_client.send_goal_async(goal_msg)
    result = await asyncio.wrap_future(future)
    return {"accepted": getattr(result, "accepted", False)}


@mcp.tool(
    name="cancel_goal",
    description="Cancel the current navigation goal",
    annotations={
        "title": "Cancel Goal",
        "readOnlyHint": True,
        "openWorldHint": False,
    },
)
async def cancel_goal() -> Dict[str, str]:
    """Cancel all currently running goals for NavigateToPose."""
    # ActionClient supports cancel_all_goals_async() on the client
    future = nav_node.nav_to_pose_client.cancel_all_goals_async()
    await asyncio.wrap_future(future)
    return {"status": "cancelled"}


# -------------------------------
# Resources (queries)
# -------------------------------


@mcp.resource(
    "mcp://nav2/get_robot_pose",
    name="get_robot_pose",
    description="Return the last robot pose estimated from odometry",
    annotations={
        "title": "Get Robot Pose",
        "readOnlyHint": True,
        "openWorldHint": True,
    },
)
def get_robot_pose() -> Dict[str, Optional[float]]:
    """Return the last odometry pose of the robot or None if not available."""
    if nav_node.pose is None:
        return {"pose": None}
    p = nav_node.pose
    return {
        "x": p.position.x,
        "y": p.position.y,
        "z": p.position.z,
        "qx": p.orientation.x,
        "qy": p.orientation.y,
        "qz": p.orientation.z,
        "qw": p.orientation.w,
    }


@mcp.resource(
    "mcp://nav2/get_navigation_status",
    name="get_navigation_status",
    description=(
        "Simplified report whether the NavigateToPose "
        "action server is ready"
    ),
    annotations={
        "title": "Navigation Status",
        "readOnlyHint": True,
        "openWorldHint": True,
    },
)
def get_navigation_status() -> Dict[str, bool]:
    """Report if the NavigateToPose action server is ready (simplified)."""
    active = nav_node.nav_to_pose_client.server_is_ready()
    return {"nav_to_pose_active": active}


async def main() -> None:
    """Run the MCP server.

    Notes
    -----
    Default transport is stdio for local MCP integration.
    """
    rclpy.init()
    await mcp.run_async(transport="stdio")

    # Clean up
    rclpy.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
