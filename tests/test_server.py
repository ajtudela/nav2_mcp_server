"""Tests for the main Nav2 MCP Server module.

This module tests the server initialization, tool registration,
and basic server functionality.
"""

from unittest.mock import Mock, patch

from fastmcp import Client

from nav2_mcp_server.server import create_server


async def test_server_initialization() -> None:
    """Test that the server initializes correctly.

    Verifies that the FastMCP server instance is created with
    the correct name and configuration.
    """
    with patch('nav2_mcp_server.server.get_config') as mock_config:
        mock_config.return_value.server.server_name = 'Nav2 MCP'

        server = create_server()
        assert server.name == 'Nav2 MCP'
        assert server is not None


async def test_server_has_tools() -> None:
    """Test that the server has tools registered.

    Verifies that tools are properly registered during server
    initialization.
    """
    with patch('nav2_mcp_server.server.get_config'):
        with patch('nav2_mcp_server.tools.get_navigation_manager'):
            with patch('nav2_mcp_server.tools.get_transform_manager'):
                server = create_server()
                tools = await server.get_tools()
                assert len(tools) > 0, (
                    'Server should have at least one tool registered'
                )


async def test_server_tool_names() -> None:
    """Test that expected tools are registered with correct names.

    Verifies that all core navigation tools are present in the server.
    """
    with patch('nav2_mcp_server.server.get_config'):
        with patch('nav2_mcp_server.tools.get_navigation_manager'):
            with patch('nav2_mcp_server.tools.get_transform_manager'):
                server = create_server()
                tools = await server.get_tools()

                # get_tools() returns a dict mapping names to tool definitions
                if isinstance(tools, dict):
                    tool_names = list(tools.keys())
                elif isinstance(tools, list):
                    tool_names = [
                        t if isinstance(t, str) else t.name for t in tools
                    ]
                else:
                    tool_names = []

                expected_tools = [
                    'navigate_to_pose',
                    'follow_waypoints',
                    'spin_robot',
                    'backup_robot',
                    'dock_robot',
                    'undock_robot',
                    'get_path',
                    'get_path_from_robot',
                    'clear_costmaps',
                    'get_robot_pose',
                    'cancel_navigation',
                    'nav2_lifecycle',
                ]

                for expected_tool in expected_tools:
                    assert expected_tool in tool_names, (
                        f"Tool '{expected_tool}' should be registered"
                    )


async def test_server_has_resources() -> None:
    """Test that the server has resources registered.

    Verifies that resources are properly registered during server
    initialization.
    """
    with patch('nav2_mcp_server.server.get_config'):
        with patch('nav2_mcp_server.resources.get_transform_manager'):
            server = create_server()
            resources = await server.get_resources()
            assert len(resources) > 0, (
                'Server should have at least one resource registered'
            )


async def test_server_client_connection() -> None:
    """Test that a client can connect to the server.

    Verifies that the in-memory transport works correctly and
    the client can establish a connection.
    """
    with patch('nav2_mcp_server.server.get_config'):
        with patch('nav2_mcp_server.tools.get_navigation_manager'):
            with patch('nav2_mcp_server.tools.get_transform_manager'):
                with patch('nav2_mcp_server.resources.get_transform_manager'):
                    server = create_server()
                    async with Client(server) as client:
                        # Test that client is connected
                        result = await client.ping()
                        assert result is True, (
                            'Client should be able to ping server'
                        )


async def test_server_list_tools_via_client() -> None:
    """Test that tools can be listed through a client connection.

    Verifies that the MCP protocol correctly exposes available tools.
    """
    with patch('nav2_mcp_server.server.get_config'):
        with patch('nav2_mcp_server.tools.get_navigation_manager'):
            with patch('nav2_mcp_server.tools.get_transform_manager'):
                with patch('nav2_mcp_server.resources.get_transform_manager'):
                    server = create_server()
                    async with Client(server) as client:
                        tools_response = await client.list_tools()
                        assert len(tools_response) > 0, (
                            'Client should receive list of available tools'
                        )


async def test_server_list_resources_via_client() -> None:
    """Test that resources can be listed through a client connection.

    Verifies that the MCP protocol correctly exposes available resources.
    """
    with patch('nav2_mcp_server.server.get_config'):
        with patch('nav2_mcp_server.tools.get_navigation_manager'):
            with patch('nav2_mcp_server.tools.get_transform_manager'):
                with patch('nav2_mcp_server.resources.get_transform_manager'):
                    server = create_server()
                    async with Client(server) as client:
                        resources_response = await client.list_resources()
                        assert len(resources_response) > 0, (
                            'Client should receive list of available resources'
                        )
