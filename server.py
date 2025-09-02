"""MCP server wrapping Nav2 action clients.

This module exposes tools and resources for navigation via FastMCP.
"""

import anyio
import asyncio
from fastmcp import FastMCP, Context
import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.duration import Duration


# Create MCP application
mcp = FastMCP('nav2-mcp-server')


# -------------------------------
# Tools (actions)
# -------------------------------

def _send_navigation_goal_sync(x: float, y: float, ctx: Context) -> str:
    """
    Synchronous helper that performs blocking Nav2 operations.

    This function is intended to run in a worker thread (via
    anyio.to_thread.run_sync). It uses anyio.from_thread.run to call the
    asynchronous ctx logging helpers from the worker thread.

    Parameters
    ----------
    x : float
        X coordinate of the target pose in the `map` frame.
    y : float
        Y coordinate of the target pose in the `map` frame.
    ctx : Context
        MCP context used for logging and progress messages (async methods).

    Returns
    -------
    str
        Human-readable status message (same semantics as the original).
    """
    navigator = BasicNavigator()

    # Report from thread into the async context
    anyio.from_thread.run(ctx.info, 'Waiting for Nav2 to become active...')
    navigator.waitUntilNav2Active()
    anyio.from_thread.run(ctx.info, 'Nav2 is active')

    goal_pose = PoseStamped()
    goal_pose.header.frame_id = 'map'
    goal_pose.header.stamp = navigator.get_clock().now().to_msg()
    goal_pose.pose.position.x = x
    goal_pose.pose.position.y = y
    goal_pose.pose.orientation.w = 1.0
    goal_pose.pose.orientation.z = 0.0

    navigator.get_logger().info(f'Sending NavigateToPose goal: x={x}, y={y}')

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
            anyio.from_thread.run(
                ctx.info,
                f'Estimated time of arrival: {seconds:.0f} seconds.'
            )

    result = navigator.getResult()
    if result == TaskResult.SUCCEEDED:
        return 'Navigation complete'
    elif result == TaskResult.FAILED:
        return 'Navigation failed'
    elif result == TaskResult.CANCELED:
        return 'Navigation canceled'
    else:
        # Handle unexpected or None results explicitly
        navigator.get_logger().warning(f'Unexpected navigation result: {result}')
        return 'Navigation unknown'


@mcp.tool(
    name='send_navigation_goal',
    description='Send a NavigateToPose goal to Nav2 at the given x, y coordinates '
                'in the map frame.',
    annotations={
        'title': 'Navigate To',
        'readOnlyHint': True,
        'openWorldHint': False,
        # LLM-friendly metadata: natural language keywords and example prompts
        'keywords': ['navigate', 'go to', 'send navigation goal', 'navigate to pose', 'move to'],
        'examplePrompts': [
            'send a navigation goal to position -6.0, 0.0',
            'go to x=-6 y=0 in the map frame',
            'navigate to (-6, 0)'
        ],
    },
)
async def send_navigation_goal(x: float, y: float, ctx: Context) -> str:
    """Send a NavigateToPose goal to Nav2.

    This tool sends a single navigation goal expressed in the `map` frame using
    the Nav2 `NavigateToPose` action via the `BasicNavigator` helper.

    The docstring intentionally includes natural language examples so that
    LLMs which inspect available tools can match user prompts such as:
    - "send a navigation goal to position -6.0, 0.0"
    - "go to x=-6 y=0 in the map frame"
    - "navigate to (-6, 0)"

    Parameters
    ----------
    x : float
        X coordinate of the target pose in the `map` frame.
    y : float
        Y coordinate of the target pose in the `map` frame.
    ctx : Context
        MCP context used for logging and progress messages.

    Returns
    -------
    str
        Human-readable status message. Possible values:
        - 'Navigation complete'
        - 'Navigation failed'
        - 'Navigation canceled'
        - 'Navigation unknown' (fallback for unexpected results)

    Notes
    -----
    - Orientation is left as a default forward-facing quaternion (w=1.0).
    - The function waits until Nav2 is active before sending the goal and
      periodically reports ETA via the MCP context.
    """
    return await anyio.to_thread.run_sync(
        _send_navigation_goal_sync, x, y, ctx
    )


async def main() -> None:
    """Run the MCP server.

    Notes
    -----
    Default transport is stdio for local MCP integration.
    """
    rclpy.init()
    await mcp.run_async(transport='stdio')

    # Clean up
    rclpy.shutdown()


if __name__ == '__main__':
    asyncio.run(main())
