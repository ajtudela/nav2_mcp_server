"""MCP resources for Nav2 navigation information.

This module provides MCP resource endpoints for accessing navigation
status and robot pose information.
"""

from fastmcp import FastMCP

from .config import get_config
from .navigation import get_navigation_manager
from .transforms import get_transform_manager
from .utils import MCPContextManager, safe_json_dumps


def create_mcp_resources(mcp: FastMCP) -> None:
    """Create and register all MCP resources with the FastMCP instance.

    Parameters
    ----------
    mcp : FastMCP
        The FastMCP server instance to register resources with.
    """
    config = get_config()

    @mcp.resource(
        uri=config.server.status_uri,
        name='Navigation Status',
        description='Current status of the navigation system',
        mime_type='application/json'
    )
    async def get_nav_status_resource() -> str:
        """Resource endpoint for navigation status."""
        try:
            nav_manager = get_navigation_manager()
            context_manager = MCPContextManager()
            status_info = nav_manager.get_navigation_status(context_manager)
            return safe_json_dumps(status_info)
        except Exception as e:
            error_info = {
                'error': 'Failed to get navigation status',
                'message': str(e),
                'error_type': type(e).__name__
            }
            return safe_json_dumps(error_info)

    @mcp.resource(
        uri=config.server.pose_uri,
        name='Robot Pose',
        description='Current robot pose in map frame',
        mime_type='application/json'
    )
    async def get_robot_pose_resource() -> str:
        """Resource endpoint for robot pose."""
        try:
            transform_manager = get_transform_manager()
            context_manager = MCPContextManager()
            pose_info = transform_manager.get_robot_pose(context_manager)
            return safe_json_dumps(pose_info)
        except Exception as e:
            error_info = {
                'error': 'Failed to get robot pose',
                'message': str(e),
                'error_type': type(e).__name__
            }
            return safe_json_dumps(error_info)
