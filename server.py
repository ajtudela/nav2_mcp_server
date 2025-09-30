"""MCP server wrapping Nav2 action clients.

This module exposes tools and resources for navigation via FastMCP.
Provides comprehensive navigation capabilities including pose navigation,
waypoint following, path planning, costmap operations, and lifecycle
management.
"""

import asyncio
from typing import Optional

import rclpy
from fastmcp import FastMCP

from .config import get_config
from .utils import setup_logging
from .mcp_tools import create_mcp_tools
from .resources import create_mcp_resources
from .navigation import get_navigation_manager
from .transforms import get_transform_manager


def create_server() -> FastMCP:
    """Create and configure the MCP server instance.

    Returns
    -------
    FastMCP
        Configured MCP server instance.
    """
    config = get_config()

    # Create MCP application
    mcp = FastMCP(config.server.server_name)

    # Register tools and resources
    create_mcp_tools(mcp)
    create_mcp_resources(mcp)

    return mcp


async def main() -> None:
    """Run the Nav2 MCP server.

    Initializes ROS2, sets up logging, and starts the MCP server
    with stdio transport for integration with MCP clients.

    Notes
    -----
    - Uses stdio transport for local MCP integration
    - Configures structured logging for debugging
    - Initializes global manager instances
    - Handles graceful shutdown of ROS2 nodes
    """
    # Setup logging
    logger = setup_logging()
    logger.info('Starting Nav2 MCP Server...')

    # Initialize ROS2
    rclpy.init()
    logger.info('ROS2 initialized')

    # Initialize global managers to check connectivity
    nav_manager: Optional[object] = None
    transform_manager: Optional[object] = None

    try:
        # Pre-initialize managers to check ROS2 connectivity
        nav_manager = get_navigation_manager()
        transform_manager = get_transform_manager()
        logger.info('Navigation and transform managers initialized')

        # Create and start MCP server
        server = create_server()
        logger.info('Starting MCP server on stdio transport')
        await server.run_async(transport='stdio')

    except KeyboardInterrupt:
        logger.info('Server interrupted by user')
    except Exception as e:
        logger.error(f'Server error: {e}')
        raise
    finally:
        # Clean up ROS2
        logger.info('Shutting down ROS2...')
        if nav_manager and hasattr(nav_manager, 'destroy'):
            nav_manager.destroy()
        if transform_manager and hasattr(transform_manager, 'destroy'):
            transform_manager.destroy()
        rclpy.shutdown()
        logger.info('Nav2 MCP Server shutdown complete')


if __name__ == '__main__':
    asyncio.run(main())
