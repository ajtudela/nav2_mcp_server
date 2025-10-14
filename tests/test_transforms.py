"""Tests for TransformManager operations.

This module tests the TransformManager class and its transform
operations including robot pose retrieval and coordinate transformations.
"""

import math
from unittest.mock import Mock, patch

import pytest

from nav2_mcp_server.transforms import TransformManager
from nav2_mcp_server.utils import MCPContextManager


class TestTransformManagerInitialization:
    """Tests for TransformManager initialization."""

    @patch('nav2_mcp_server.transforms.rclpy.create_node')
    @patch('nav2_mcp_server.transforms.Buffer')
    @patch('nav2_mcp_server.transforms.TransformListener')
    def test_transform_manager_init(
        self, mock_listener, mock_buffer, mock_create_node
    ):
        """Test TransformManager initialization.

        Verifies that the TransformManager properly initializes
        with TF2 components.
        """
        mock_node = Mock()
        mock_create_node.return_value = mock_node
        mock_buffer_instance = Mock()
        mock_buffer.return_value = mock_buffer_instance

        tf_manager = TransformManager()

        assert tf_manager is not None

    @patch('nav2_mcp_server.transforms.rclpy.create_node')
    @patch('nav2_mcp_server.transforms.Buffer')
    @patch('nav2_mcp_server.transforms.TransformListener')
    def test_transform_manager_tf_setup(
        self, mock_listener, mock_buffer, mock_create_node
    ):
        """Test TransformManager TF setup.

        Verifies that TF2 buffer and listener are properly configured.
        """
        mock_node = Mock()
        mock_create_node.return_value = mock_node
        mock_buffer_instance = Mock()
        mock_buffer.return_value = mock_buffer_instance

        tf_manager = TransformManager()
        tf_manager._ensure_tf_setup()

        mock_buffer.assert_called()
        mock_listener.assert_called()


class TestGetRobotPose:
    """Tests for robot pose retrieval functionality."""

    @patch('nav2_mcp_server.transforms.rclpy.create_node')
    @patch('nav2_mcp_server.transforms.Buffer')
    @patch('nav2_mcp_server.transforms.TransformListener')
    def test_get_robot_pose_success(
        self, mock_listener, mock_buffer, mock_create_node
    ):
        """Test successful robot pose retrieval.

        Verifies that robot pose is correctly retrieved from TF2.
        """
        # Setup mocks
        mock_node = Mock()
        mock_create_node.return_value = mock_node
        mock_buffer_instance = Mock()
        mock_buffer.return_value = mock_buffer_instance

        # Mock transform
        mock_transform = Mock()
        mock_transform.transform.translation.x = 1.0
        mock_transform.transform.translation.y = 2.0
        mock_transform.transform.translation.z = 0.0
        mock_transform.transform.rotation.x = 0.0
        mock_transform.transform.rotation.y = 0.0
        mock_transform.transform.rotation.z = 0.0
        mock_transform.transform.rotation.w = 1.0
        mock_transform.header.frame_id = 'map'
        mock_transform.child_frame_id = 'base_link'

        mock_buffer_instance.lookup_transform.return_value = mock_transform

        tf_manager = TransformManager()
        context_manager = MCPContextManager()

        result = tf_manager.get_robot_pose(context_manager)

        assert 'pose' in result
        assert 'status' in result
        assert result['status'] == 'success'
        assert result['pose']['position']['x'] == 1.0
        assert result['pose']['position']['y'] == 2.0

    @patch('nav2_mcp_server.transforms.rclpy.create_node')
    @patch('nav2_mcp_server.transforms.Buffer')
    @patch('nav2_mcp_server.transforms.TransformListener')
    def test_get_robot_pose_transform_failure(
        self, mock_listener, mock_buffer, mock_create_node
    ):
        """Test robot pose retrieval with transform failure.

        Verifies that transform lookup failures are handled gracefully.
        """
        # Setup mocks
        mock_node = Mock()
        mock_create_node.return_value = mock_node
        mock_buffer_instance = Mock()
        mock_buffer.return_value = mock_buffer_instance

        # Mock transform failure
        mock_buffer_instance.lookup_transform.side_effect = Exception(
            'Transform not available'
        )

        tf_manager = TransformManager()
        context_manager = MCPContextManager()

        result = tf_manager.get_robot_pose(context_manager)

        assert 'error' in result
        assert result['status'] == 'error'

    @patch('nav2_mcp_server.transforms.rclpy.create_node')
    @patch('nav2_mcp_server.transforms.Buffer')
    @patch('nav2_mcp_server.transforms.TransformListener')
    def test_get_robot_pose_with_custom_frames(
        self, mock_listener, mock_buffer, mock_create_node
    ):
        """Test robot pose retrieval with custom frame names.

        Verifies that custom target and source frames work correctly.
        """
        # Setup mocks
        mock_node = Mock()
        mock_create_node.return_value = mock_node
        mock_buffer_instance = Mock()
        mock_buffer.return_value = mock_buffer_instance

        # Mock transform
        mock_transform = Mock()
        mock_transform.transform.translation.x = 3.0
        mock_transform.transform.translation.y = 4.0
        mock_transform.transform.translation.z = 0.5
        mock_transform.transform.rotation.x = 0.0
        mock_transform.transform.rotation.y = 0.0
        mock_transform.transform.rotation.z = 0.707
        mock_transform.transform.rotation.w = 0.707
        mock_transform.header.frame_id = 'odom'
        mock_transform.child_frame_id = 'robot_base'

        mock_buffer_instance.lookup_transform.return_value = mock_transform

        tf_manager = TransformManager()
        context_manager = MCPContextManager()

        result = tf_manager.get_robot_pose(
            context_manager, 
            target_frame='odom',
            source_frame='robot_base'
        )

        assert result['pose']['position']['x'] == 3.0
        assert result['pose']['position']['y'] == 4.0
        assert result['pose']['position']['z'] == 0.5


class TestPoseExtraction:
    """Tests for pose extraction from transforms."""

    @patch('nav2_mcp_server.transforms.rclpy.create_node')
    @patch('nav2_mcp_server.transforms.Buffer')
    @patch('nav2_mcp_server.transforms.TransformListener')
    def test_extract_pose_from_transform(
        self, mock_listener, mock_buffer, mock_create_node
    ):
        """Test pose extraction from transform message.

        Verifies that transform messages are correctly converted
        to pose dictionaries.
        """
        # Setup mocks
        mock_node = Mock()
        mock_create_node.return_value = mock_node

        tf_manager = TransformManager()

        # Create mock transform
        mock_transform = Mock()
        mock_transform.transform.translation.x = 1.5
        mock_transform.transform.translation.y = -2.3
        mock_transform.transform.translation.z = 0.1
        mock_transform.transform.rotation.x = 0.0
        mock_transform.transform.rotation.y = 0.0
        mock_transform.transform.rotation.z = 0.383
        mock_transform.transform.rotation.w = 0.924

        result = tf_manager._extract_pose_from_transform(mock_transform)

        assert 'position' in result
        assert 'orientation' in result
        assert result['position']['x'] == 1.5
        assert result['position']['y'] == -2.3
        assert result['position']['z'] == 0.1
        assert result['orientation']['z'] == 0.383
        assert result['orientation']['w'] == 0.924


class TestQuaternionOperations:
    """Tests for quaternion operations."""

    @patch('nav2_mcp_server.transforms.rclpy.create_node')
    def test_quaternion_to_yaw_conversion(self, mock_create_node):
        """Test quaternion to yaw angle conversion.

        Verifies that quaternions are correctly converted to yaw angles.
        """
        mock_node = Mock()
        mock_create_node.return_value = mock_node

        tf_manager = TransformManager()

        # Test known quaternion conversions
        # Identity quaternion (0 degrees)
        yaw_0 = tf_manager._quaternion_to_yaw(0.0, 0.0, 0.0, 1.0)
        assert abs(yaw_0) < 0.001

        # 90 degrees rotation
        yaw_90 = tf_manager._quaternion_to_yaw(0.0, 0.0, 0.707, 0.707)
        assert abs(yaw_90 - math.pi/2) < 0.01

        # 180 degrees rotation
        yaw_180 = tf_manager._quaternion_to_yaw(0.0, 0.0, 1.0, 0.0)
        assert abs(abs(yaw_180) - math.pi) < 0.01

    @patch('nav2_mcp_server.transforms.rclpy.create_node')
    def test_yaw_to_quaternion_conversion(self, mock_create_node):
        """Test yaw angle to quaternion conversion.

        Verifies that yaw angles are correctly converted to quaternions.
        """
        mock_node = Mock()
        mock_create_node.return_value = mock_node

        tf_manager = TransformManager()

        # Test known angle conversions
        # 0 degrees
        quat_0 = tf_manager.yaw_to_quaternion(0.0)
        assert abs(quat_0['w'] - 1.0) < 0.001
        assert abs(quat_0['z']) < 0.001

        # 90 degrees
        quat_90 = tf_manager.yaw_to_quaternion(math.pi/2)
        assert abs(quat_90['z'] - 0.707) < 0.01
        assert abs(quat_90['w'] - 0.707) < 0.01

        # 180 degrees
        quat_180 = tf_manager.yaw_to_quaternion(math.pi)
        assert abs(quat_180['w']) < 0.01
        assert abs(abs(quat_180['z']) - 1.0) < 0.01

    @patch('nav2_mcp_server.transforms.rclpy.create_node')
    def test_yaw_quaternion_roundtrip(self, mock_create_node):
        """Test yaw-quaternion-yaw conversion roundtrip.

        Verifies that converting yaw to quaternion and back
        preserves the original value.
        """
        mock_node = Mock()
        mock_create_node.return_value = mock_node

        tf_manager = TransformManager()

        test_angles = [0.0, math.pi/4, math.pi/2, math.pi, -math.pi/2]

        for original_yaw in test_angles:
            # Convert to quaternion and back
            quat = tf_manager.yaw_to_quaternion(original_yaw)
            recovered_yaw = tf_manager._quaternion_to_yaw(
                quat['x'], quat['y'], quat['z'], quat['w']
            )

            # Account for angle wrapping
            diff = abs(original_yaw - recovered_yaw)
            if diff > math.pi:
                diff = 2 * math.pi - diff

            assert diff < 0.01, f'Roundtrip failed for {original_yaw}'


class TestTransformManagerDestroy:
    """Tests for TransformManager cleanup."""

    @patch('nav2_mcp_server.transforms.rclpy.create_node')
    @patch('nav2_mcp_server.transforms.Buffer')
    @patch('nav2_mcp_server.transforms.TransformListener')
    def test_transform_manager_destroy(
        self, mock_listener, mock_buffer, mock_create_node
    ):
        """Test TransformManager cleanup.

        Verifies that the TransformManager properly cleans up
        ROS2 resources on destruction.
        """
        mock_node = Mock()
        mock_create_node.return_value = mock_node

        tf_manager = TransformManager()
        tf_manager.destroy()

        # Verify node cleanup is called
        mock_node.destroy_node.assert_called_once()


class TestTransformManagerErrorHandling:
    """Tests for TransformManager error handling."""

    @patch('nav2_mcp_server.transforms.rclpy.create_node')
    @patch('nav2_mcp_server.transforms.Buffer')
    @patch('nav2_mcp_server.transforms.TransformListener')
    def test_get_robot_pose_timeout_error(
        self, mock_listener, mock_buffer, mock_create_node
    ):
        """Test robot pose retrieval with timeout error.

        Verifies that timeout errors are handled appropriately.
        """
        mock_node = Mock()
        mock_create_node.return_value = mock_node
        mock_buffer_instance = Mock()
        mock_buffer.return_value = mock_buffer_instance

        # Mock timeout exception
        from tf2_ros import LookupException
        mock_buffer_instance.lookup_transform.side_effect = LookupException(
            'Transform timeout'
        )

        tf_manager = TransformManager()
        context_manager = MCPContextManager()

        result = tf_manager.get_robot_pose(context_manager)

        assert 'error' in result
        assert result['status'] == 'error'
        assert 'timeout' in result['message'].lower()

    @patch('nav2_mcp_server.transforms.rclpy.create_node')
    def test_transform_manager_init_failure(self, mock_create_node):
        """Test TransformManager initialization failure.

        Verifies that initialization failures are handled gracefully.
        """
        mock_create_node.side_effect = Exception('Node creation failed')

        with pytest.raises(Exception):
            TransformManager()