"""Tests for Nav2 MCP Server resources.

This module tests all resource endpoints including robot pose
information and error handling.
"""

from unittest.mock import Mock, patch

from fastmcp import Client

from nav2_mcp_server.server import create_server


class TestRobotPoseResource:
    """Tests for robot pose resource endpoint."""

    async def test_robot_pose_resource_success(
        self,
        mock_transform_manager: Mock
    ):
        """Test successful robot pose resource retrieval.

        Verifies that the resource endpoint correctly returns
        robot pose information in JSON format.
        """
        with patch('nav2_mcp_server.server.get_config'):
            with patch('nav2_mcp_server.tools.get_navigation_manager'):
                with patch('nav2_mcp_server.tools.get_transform_manager'):
                    with patch(
                        'nav2_mcp_server.resources.get_transform_manager',
                        return_value=mock_transform_manager
                    ):
                        server = create_server()
                        async with Client(server) as client:
                            resources = await client.list_resources()
                            assert len(resources) > 0

                            # Get the robot pose resource
                            pose_resource = None
                            for resource in resources:
                                if 'pose' in resource.uri.lower():
                                    pose_resource = resource
                                    break

                            assert pose_resource is not None
                            assert pose_resource.mime_type == 'application/json'

    async def test_robot_pose_resource_content(
        self,
        mock_transform_manager: Mock
    ):
        """Test robot pose resource returns correct content.

        Verifies that the resource content matches expected
        pose data structure.
        """
        expected_pose = {
            'pose': {
                'position': {'x': 1.0, 'y': 2.0, 'z': 0.0},
                'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
            },
            'header': {
                'frame_id': 'map',
                'stamp': {'sec': 1640995200, 'nanosec': 0}
            },
            'yaw': 0.0,
            'status': 'success'
        }

        mock_transform_manager.get_robot_pose.return_value = expected_pose

        with patch('nav2_mcp_server.server.get_config') as mock_config:
            mock_config.return_value.server.pose_uri = 'nav2://robot_pose'
            with patch('nav2_mcp_server.tools.get_navigation_manager'):
                with patch('nav2_mcp_server.tools.get_transform_manager'):
                    with patch(
                        'nav2_mcp_server.resources.get_transform_manager',
                        return_value=mock_transform_manager
                    ):
                        server = create_server()
                        async with Client(server) as client:
                            # Get resource content
                            resource_content = await client.read_resource(
                                'nav2://robot_pose'
                            )

                            assert resource_content is not None
                            # Content should be in JSON format
                            import json
                            parsed_content = json.loads(resource_content.content)
                            assert 'pose' in parsed_content
                            assert 'status' in parsed_content

    async def test_robot_pose_resource_error_handling(self):
        """Test robot pose resource handles errors gracefully.

        Verifies that the resource returns error information
        when transform lookup fails.
        """
        mock_tf_manager = Mock()
        mock_tf_manager.get_robot_pose.side_effect = Exception(
            'Transform lookup failed'
        )

        with patch('nav2_mcp_server.server.get_config') as mock_config:
            mock_config.return_value.server.pose_uri = 'nav2://robot_pose'
            with patch('nav2_mcp_server.tools.get_navigation_manager'):
                with patch('nav2_mcp_server.tools.get_transform_manager'):
                    with patch(
                        'nav2_mcp_server.resources.get_transform_manager',
                        return_value=mock_tf_manager
                    ):
                        server = create_server()
                        async with Client(server) as client:
                            # Get resource content with error
                            resource_content = await client.read_resource(
                                'nav2://robot_pose'
                            )

                            assert resource_content is not None
                            # Content should contain error information
                            import json
                            parsed_content = json.loads(resource_content.content)
                            assert 'error' in parsed_content
                            assert 'message' in parsed_content

    async def test_robot_pose_resource_mime_type(self):
        """Test robot pose resource has correct MIME type.

        Verifies that the resource is properly configured
        with JSON MIME type.
        """
        with patch('nav2_mcp_server.server.get_config'):
            with patch('nav2_mcp_server.tools.get_navigation_manager'):
                with patch('nav2_mcp_server.tools.get_transform_manager'):
                    with patch(
                        'nav2_mcp_server.resources.get_transform_manager'
                    ):
                        server = create_server()
                        async with Client(server) as client:
                            resources = await client.list_resources()

                            # Find the robot pose resource
                            pose_resource = None
                            for resource in resources:
                                if 'robot pose' in resource.name.lower():
                                    pose_resource = resource
                                    break

                            assert pose_resource is not None
                            assert pose_resource.mime_type == 'application/json'
                            assert 'robot pose' in pose_resource.description.lower()


class TestResourceErrorHandling:
    """Tests for resource error handling scenarios."""

    async def test_resource_with_manager_unavailable(self):
        """Test resource behavior when manager is unavailable.

        Verifies that resources handle manager initialization
        failures gracefully.
        """
        with patch('nav2_mcp_server.server.get_config'):
            with patch('nav2_mcp_server.tools.get_navigation_manager'):
                with patch('nav2_mcp_server.tools.get_transform_manager'):
                    with patch(
                        'nav2_mcp_server.resources.get_transform_manager',
                        side_effect=Exception('Manager unavailable')
                    ):
                        server = create_server()
                        async with Client(server) as client:
                            # Should still be able to list resources
                            resources = await client.list_resources()
                            assert isinstance(resources, list)

    async def test_resource_json_serialization(self):
        """Test resource handles complex data serialization.

        Verifies that the resource can handle various data types
        in the pose information.
        """
        mock_tf_manager = Mock()
        complex_pose = {
            'pose': {
                'position': {'x': 1.5, 'y': -2.3, 'z': 0.1},
                'orientation': {
                    'x': 0.0, 'y': 0.0, 'z': 0.707, 'w': 0.707
                }
            },
            'header': {
                'frame_id': 'map',
                'stamp': {'sec': 1640995200, 'nanosec': 123456789}
            },
            'yaw': 1.5708,
            'status': 'success',
            'additional_info': {
                'confidence': 0.95,
                'source': 'amcl'
            }
        }

        mock_tf_manager.get_robot_pose.return_value = complex_pose

        with patch('nav2_mcp_server.server.get_config') as mock_config:
            mock_config.return_value.server.pose_uri = 'nav2://robot_pose'
            with patch('nav2_mcp_server.tools.get_navigation_manager'):
                with patch('nav2_mcp_server.tools.get_transform_manager'):
                    with patch(
                        'nav2_mcp_server.resources.get_transform_manager',
                        return_value=mock_tf_manager
                    ):
                        server = create_server()
                        async with Client(server) as client:
                            # Get resource content
                            resource_content = await client.read_resource(
                                'nav2://robot_pose'
                            )

                            assert resource_content is not None
                            # Should be able to parse complex JSON
                            import json
                            parsed_content = json.loads(resource_content.content)
                            assert parsed_content['yaw'] == 1.5708
                            assert 'additional_info' in parsed_content
                            assert parsed_content['additional_info']['confidence'] == 0.95
