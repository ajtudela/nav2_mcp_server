"""Tests for the main Nav2 MCP Server module.

This module tests the server initialization, tool registration,
and basic server functionality.
"""

from unittest.mock import AsyncMock, Mock, patch

from fastmcp import Client

from nav2_mcp_server.server import create_server


@patch('nav2_mcp_server.server.get_config')
async def test_server_initialization(mock_config: Mock) -> None:
    """Test that the server initializes correctly.

    Verifies that the FastMCP server instance is created with
    the correct name and configuration.
    """
    mock_config.return_value.server.server_name = 'Nav2 MCP'

    server = create_server()
    assert server.name == 'Nav2 MCP'
    assert server is not None


@patch('nav2_mcp_server.tools.get_transform_manager')
@patch('nav2_mcp_server.tools.get_navigation_manager')
@patch('nav2_mcp_server.server.get_config')
async def test_server_has_tools(
    mock_server_get_config: Mock,
    mock_get_nav_manager: Mock,
    mock_get_tf_manager: Mock
) -> None:
    """Test that the server has tools registered.

    Verifies that tools are properly registered during server
    initialization.
    """
    server = create_server()
    tools = await server.get_tools()
    assert len(tools) > 0, (
        'Server should have at least one tool registered'
    )


@patch('nav2_mcp_server.tools.get_transform_manager')
@patch('nav2_mcp_server.tools.get_navigation_manager')
@patch('nav2_mcp_server.server.get_config')
async def test_server_tool_names(
    mock_server_get_config: Mock,
    mock_get_nav_manager: Mock,
    mock_get_tf_manager: Mock
) -> None:
    """Test that expected tools are registered with correct names.

    Verifies that all core navigation tools are present in the server.
    """
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


@patch('nav2_mcp_server.resources.get_transform_manager')
@patch('nav2_mcp_server.server.get_config')
async def test_server_has_resources(
    mock_server_get_config: Mock,
    mock_resources_get_tf_manager: Mock
) -> None:
    """Test that the server has resources registered.

    Verifies that resources are properly registered during server
    initialization.
    """
    server = create_server()
    resources = await server.get_resources()
    assert len(resources) > 0, (
        'Server should have at least one resource registered'
    )


@patch('nav2_mcp_server.resources.get_transform_manager')
@patch('nav2_mcp_server.tools.get_transform_manager')
@patch('nav2_mcp_server.tools.get_navigation_manager')
@patch('nav2_mcp_server.server.get_config')
async def test_server_client_connection(
    mock_server_get_config: Mock,
    mock_get_nav_manager: Mock,
    mock_get_tf_manager: Mock,
    mock_resources_get_tf_manager: Mock
) -> None:
    """Test that a client can connect to the server.

    Verifies that the in-memory transport works correctly and
    the client can establish a connection.
    """
    from tests.conftest import create_mock_config

    mock_server_get_config.return_value = create_mock_config()
    server = create_server()
    async with Client(server) as client:
        # Test that client is connected
        result = await client.ping()
        assert result is True, (
            'Client should be able to ping server'
        )


@patch('nav2_mcp_server.resources.get_transform_manager')
@patch('nav2_mcp_server.tools.get_transform_manager')
@patch('nav2_mcp_server.tools.get_navigation_manager')
@patch('nav2_mcp_server.server.get_config')
async def test_server_list_tools_via_client(
    mock_server_get_config: Mock,
    mock_get_nav_manager: Mock,
    mock_get_tf_manager: Mock,
    mock_resources_get_tf_manager: Mock
) -> None:
    """Test that tools can be listed through a client connection.

    Verifies that the MCP protocol correctly exposes available tools.
    """
    from tests.conftest import create_mock_config

    mock_server_get_config.return_value = create_mock_config()
    server = create_server()
    async with Client(server) as client:
        tools_response = await client.list_tools()
        assert len(tools_response) > 0, (
            'Client should receive list of available tools'
        )


@patch('nav2_mcp_server.resources.get_transform_manager')
@patch('nav2_mcp_server.tools.get_transform_manager')
@patch('nav2_mcp_server.tools.get_navigation_manager')
@patch('nav2_mcp_server.server.get_config')
async def test_server_list_resources_via_client(
    mock_server_get_config: Mock,
    mock_get_nav_manager: Mock,
    mock_get_tf_manager: Mock,
    mock_resources_get_tf_manager: Mock
) -> None:
    """Test that resources can be listed through a client connection.

    Verifies that the MCP protocol correctly exposes available resources.
    """
    from tests.conftest import create_mock_config

    mock_server_get_config.return_value = create_mock_config()
    server = create_server()
    async with Client(server) as client:
        resources_response = await client.list_resources()
        assert len(resources_response) > 0, (
            'Client should receive list of available resources'
        )


@patch('nav2_mcp_server.server.rclpy.shutdown')
@patch('nav2_mcp_server.server.create_server')
@patch('nav2_mcp_server.server.get_transform_manager')
@patch('nav2_mcp_server.server.get_navigation_manager')
@patch('nav2_mcp_server.server.rclpy.init')
@patch('nav2_mcp_server.server.setup_logging')
async def test_main_function_success(
    mock_setup_logging: Mock,
    mock_init: Mock,
    mock_get_nav_manager: Mock,
    mock_get_tf_manager: Mock,
    mock_create_server: Mock,
    mock_shutdown: Mock
) -> None:
    """Test the main function executes successfully.

    Verifies that the main function initializes ROS2, creates managers,
    and starts the server.
    """
    from nav2_mcp_server.server import main

    mock_server = Mock()
    mock_server.run_async = AsyncMock()
    mock_nav_manager = Mock()
    mock_nav_manager.destroy = Mock()
    mock_transform_manager = Mock()
    mock_transform_manager.destroy = Mock()
    mock_logger = Mock()

    mock_setup_logging.return_value = mock_logger
    mock_get_nav_manager.return_value = mock_nav_manager
    mock_get_tf_manager.return_value = mock_transform_manager
    mock_create_server.return_value = mock_server

    # Run main
    await main()

    # Verify initialization
    mock_logger.info.assert_any_call(
        'Starting Nav2 MCP Server...'
    )
    mock_init.assert_called_once()
    mock_logger.info.assert_any_call(
        'ROS2 initialized'
    )
    mock_logger.info.assert_any_call(
        'Navigation and transform managers initialized'
    )
    mock_logger.info.assert_any_call(
        'Starting MCP server on stdio transport'
    )
    mock_logger.info.assert_any_call(
        'Shutting down ROS2...'
    )
    mock_nav_manager.destroy.assert_called_once()
    mock_transform_manager.destroy.assert_called_once()
    mock_shutdown.assert_called_once()
    mock_logger.info.assert_any_call(
        'Nav2 MCP Server shutdown complete'
    )


@patch('nav2_mcp_server.server.rclpy.shutdown')
@patch('nav2_mcp_server.server.create_server')
@patch('nav2_mcp_server.server.get_transform_manager')
@patch('nav2_mcp_server.server.get_navigation_manager')
@patch('nav2_mcp_server.server.rclpy.init')
@patch('nav2_mcp_server.server.setup_logging')
async def test_main_function_keyboard_interrupt(
    mock_setup_logging: Mock,
    mock_init: Mock,
    mock_get_nav_manager: Mock,
    mock_get_tf_manager: Mock,
    mock_create_server: Mock,
    mock_shutdown: Mock
) -> None:
    """Test the main function handles KeyboardInterrupt gracefully.

    Verifies that the server shuts down properly when interrupted.
    """
    from nav2_mcp_server.server import main

    mock_server = Mock()
    mock_server.run_async = AsyncMock(side_effect=KeyboardInterrupt())
    mock_nav_manager = Mock()
    mock_nav_manager.destroy = Mock()
    mock_transform_manager = Mock()
    mock_transform_manager.destroy = Mock()
    mock_logger = Mock()

    mock_setup_logging.return_value = mock_logger
    mock_get_nav_manager.return_value = mock_nav_manager
    mock_get_tf_manager.return_value = mock_transform_manager
    mock_create_server.return_value = mock_server

    # Run main
    await main()

    # Verify interrupt handling
    mock_logger.info.assert_any_call(
        'Server interrupted by user'
    )
    mock_shutdown.assert_called_once()


@patch('nav2_mcp_server.server.rclpy.shutdown')
@patch('nav2_mcp_server.server.create_server')
@patch('nav2_mcp_server.server.get_transform_manager')
@patch('nav2_mcp_server.server.get_navigation_manager')
@patch('nav2_mcp_server.server.rclpy.init')
@patch('nav2_mcp_server.server.setup_logging')
async def test_main_function_exception(
    mock_setup_logging: Mock,
    mock_init: Mock,
    mock_get_nav_manager: Mock,
    mock_get_tf_manager: Mock,
    mock_create_server: Mock,
    mock_shutdown: Mock
) -> None:
    """Test the main function handles exceptions properly.

    Verifies that exceptions are logged and re-raised.
    """
    import pytest

    from nav2_mcp_server.server import main

    mock_server = Mock()
    mock_server.run_async = AsyncMock(side_effect=RuntimeError('Test error'))
    mock_nav_manager = Mock()
    mock_nav_manager.destroy = Mock()
    mock_transform_manager = Mock()
    mock_transform_manager.destroy = Mock()
    mock_logger = Mock()

    mock_setup_logging.return_value = mock_logger
    mock_get_nav_manager.return_value = mock_nav_manager
    mock_get_tf_manager.return_value = mock_transform_manager
    mock_create_server.return_value = mock_server

    # Run main and expect exception
    with pytest.raises(
        RuntimeError, match='Test error'
    ):
        await main()

    # Verify error handling
    mock_logger.error.assert_any_call(
        'Server error: Test error'
    )
    mock_shutdown.assert_called_once()


@patch('nav2_mcp_server.server.rclpy.shutdown')
@patch('nav2_mcp_server.server.create_server')
@patch('nav2_mcp_server.server.get_transform_manager')
@patch('nav2_mcp_server.server.get_navigation_manager')
@patch('nav2_mcp_server.server.rclpy.init')
@patch('nav2_mcp_server.server.setup_logging')
async def test_main_function_cleanup_without_destroy(
    mock_setup_logging: Mock,
    mock_init: Mock,
    mock_get_nav_manager: Mock,
    mock_get_tf_manager: Mock,
    mock_create_server: Mock,
    mock_shutdown: Mock
) -> None:
    """Test the main function cleanup when managers don't have destroy method.

    Verifies that cleanup works even if managers lack destroy method.
    """
    from nav2_mcp_server.server import main

    mock_server = Mock()
    mock_server.run_async = AsyncMock()
    # Create managers without destroy method
    mock_nav_manager = Mock(spec=[])
    mock_transform_manager = Mock(spec=[])
    mock_logger = Mock()

    mock_setup_logging.return_value = mock_logger
    mock_get_nav_manager.return_value = mock_nav_manager
    mock_get_tf_manager.return_value = mock_transform_manager
    mock_create_server.return_value = mock_server

    # Run main
    await main()

    # Verify cleanup still works
    mock_shutdown.assert_called_once()
    mock_logger.info.assert_any_call(
        'Nav2 MCP Server shutdown complete'
    )
