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

    def test_transform_manager_init(self) -> None:
        """Test TransformManager initialization.

        Verifies that the TransformManager properly initializes.
        """
        tf_manager = TransformManager()

        assert tf_manager is not None
        assert tf_manager._node is None  # Node not created until needed

    @patch('nav2_mcp_server.transforms.rclpy.spin_once')
    @patch('nav2_mcp_server.transforms.TransformListener')
    @patch('nav2_mcp_server.transforms.Buffer')
    @patch('nav2_mcp_server.transforms.Node')
    @patch('nav2_mcp_server.transforms.rclpy.init')
    def test_transform_manager_tf_setup(
        self,
        mock_init: Mock,
        mock_node_class: Mock,
        mock_buffer_class: Mock,
        mock_listener_class: Mock,
        mock_spin: Mock
    ) -> None:
        """Test TransformManager TF setup.

        Verifies that TF2 buffer and listener are properly configured.
        """
        mock_node = Mock()
        mock_node_class.return_value = mock_node
        mock_buffer_instance = Mock()
        mock_buffer_class.return_value = mock_buffer_instance

        # Initialize rclpy for the test
        mock_init.return_value = None

        tf_manager = TransformManager()
        tf_manager._ensure_tf_setup()

        mock_node_class.assert_called_once()
        mock_buffer_class.assert_called_once()
        mock_listener_class.assert_called_once()


class TestGetRobotPose:
    """Tests for robot pose retrieval functionality."""

    @patch('nav2_mcp_server.transforms.rclpy.spin_once')
    @patch('nav2_mcp_server.transforms.TransformListener')
    @patch('nav2_mcp_server.transforms.Buffer')
    @patch('nav2_mcp_server.transforms.Node')
    def test_get_robot_pose_success(
        self,
        mock_node_class: Mock,
        mock_buffer_class: Mock,
        mock_listener_class: Mock,
        mock_spin: Mock
    ) -> None:
        """Test successful robot pose retrieval.

        Verifies that robot pose is correctly retrieved from TF2.
        """
        # Setup mocks
        mock_node = Mock()
        mock_node_class.return_value = mock_node
        mock_buffer_instance = Mock()
        mock_buffer_class.return_value = mock_buffer_instance

        # Mock transform
        mock_transform = Mock()
        mock_transform.transform.translation.x = 1.0
        mock_transform.transform.translation.y = 2.0
        mock_transform.transform.translation.z = 0.0
        mock_transform.transform.rotation.x = 0.0
        mock_transform.transform.rotation.y = 0.0
        mock_transform.transform.rotation.z = 0.0
        mock_transform.transform.rotation.w = 1.0
        mock_transform.header.stamp.sec = 100
        mock_transform.header.stamp.nanosec = 0

        mock_buffer_instance.can_transform.return_value = True
        mock_buffer_instance.lookup_transform.return_value = mock_transform

        tf_manager = TransformManager()
        context_manager = MCPContextManager()

        result = tf_manager.get_robot_pose(context_manager)

        assert 'position' in result
        assert 'orientation' in result
        assert result['position']['x'] == 1.0
        assert result['position']['y'] == 2.0

    @patch('nav2_mcp_server.transforms.rclpy.spin_once')
    @patch('nav2_mcp_server.transforms.TransformListener')
    @patch('nav2_mcp_server.transforms.Buffer')
    @patch('nav2_mcp_server.transforms.Node')
    def test_get_robot_pose_transform_failure(
        self,
        mock_node_class: Mock,
        mock_buffer_class: Mock,
        mock_listener_class: Mock,
        mock_spin: Mock
    ) -> None:
        """Test robot pose retrieval with transform failure.

        Verifies that transform lookup failures are handled gracefully.
        """
        from nav2_mcp_server.exceptions import TransformError

        # Setup mocks
        mock_node = Mock()
        mock_node_class.return_value = mock_node
        mock_buffer_instance = Mock()
        mock_buffer_class.return_value = mock_buffer_instance

        # Mock transform failure
        mock_buffer_instance.can_transform.return_value = True
        mock_buffer_instance.lookup_transform.side_effect = Exception(
            'Transform not available'
        )

        tf_manager = TransformManager()
        context_manager = MCPContextManager()

        # Should raise TransformError
        with pytest.raises(TransformError):
            tf_manager.get_robot_pose(context_manager)

    @patch('nav2_mcp_server.transforms.rclpy.spin_once')
    @patch('nav2_mcp_server.transforms.TransformListener')
    @patch('nav2_mcp_server.transforms.Buffer')
    @patch('nav2_mcp_server.transforms.Node')
    def test_get_robot_pose_with_custom_frames(
        self,
        mock_node_class: Mock,
        mock_buffer_class: Mock,
        mock_listener_class: Mock,
        mock_spin: Mock
    ) -> None:
        """Test robot pose retrieval works with config frames.

        Verifies that the method uses frames from config.
        """
        # Setup mocks
        mock_node = Mock()
        mock_node_class.return_value = mock_node
        mock_buffer_instance = Mock()
        mock_buffer_class.return_value = mock_buffer_instance

        # Mock transform
        mock_transform = Mock()
        mock_transform.transform.translation.x = 3.0
        mock_transform.transform.translation.y = 4.0
        mock_transform.transform.translation.z = 0.5
        mock_transform.transform.rotation.x = 0.0
        mock_transform.transform.rotation.y = 0.0
        mock_transform.transform.rotation.z = 0.707
        mock_transform.transform.rotation.w = 0.707
        mock_transform.header.stamp.sec = 100
        mock_transform.header.stamp.nanosec = 0

        mock_buffer_instance.can_transform.return_value = True
        mock_buffer_instance.lookup_transform.return_value = mock_transform

        tf_manager = TransformManager()
        context_manager = MCPContextManager()

        # Note: get_robot_pose doesn't take frame arguments, uses config
        result = tf_manager.get_robot_pose(context_manager)

        assert result['position']['x'] == 3.0
        assert result['position']['y'] == 4.0
        assert result['position']['z'] == 0.5


class TestPoseExtraction:
    """Tests for pose extraction from transforms."""

    def test_extract_pose_from_transform(self) -> None:
        """Test pose extraction from transform message.

        Verifies that transform messages are correctly converted
        to pose dictionaries.
        """
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
        mock_transform.header.stamp.sec = 100
        mock_transform.header.stamp.nanosec = 0

        result = tf_manager._extract_pose_from_transform(mock_transform)

        assert 'position' in result
        assert 'orientation' in result
        assert result['position']['x'] == 1.5
        assert result['position']['y'] == -2.3
        assert result['position']['z'] == 0.1
        # Check quaternion inside orientation
        assert result['orientation']['quaternion']['z'] == 0.383
        assert result['orientation']['quaternion']['w'] == 0.924


class TestQuaternionOperations:
    """Tests for quaternion operations."""

    def test_quaternion_to_yaw_conversion(self) -> None:
        """Test quaternion to yaw angle conversion.

        Verifies that quaternions are correctly converted to yaw angles.
        """
        tf_manager = TransformManager()

        # Test known quaternion conversions
        # Identity quaternion (0 degrees) - w=1, others=0
        yaw_0 = tf_manager._quaternion_to_yaw(1.0, 0.0, 0.0, 0.0)
        assert abs(yaw_0) < 0.001

        # 90 degrees rotation
        yaw_90 = tf_manager._quaternion_to_yaw(0.707, 0.0, 0.0, 0.707)
        assert abs(yaw_90 - math.pi / 2) < 0.01

        # 180 degrees rotation
        yaw_180 = tf_manager._quaternion_to_yaw(0.0, 0.0, 0.0, 1.0)
        assert abs(abs(yaw_180) - math.pi) < 0.01

    def test_yaw_to_quaternion_conversion(self) -> None:
        """Test yaw angle to quaternion conversion.

        Verifies that yaw angles are correctly converted to quaternions.
        """
        tf_manager = TransformManager()

        # Test known angle conversions
        # 0 degrees
        quat_0 = tf_manager.yaw_to_quaternion(0.0)
        assert abs(quat_0['w'] - 1.0) < 0.001
        assert abs(quat_0['z']) < 0.001

        # 90 degrees
        quat_90 = tf_manager.yaw_to_quaternion(math.pi / 2)
        assert abs(quat_90['z'] - 0.707) < 0.01
        assert abs(quat_90['w'] - 0.707) < 0.01

        # 180 degrees
        quat_180 = tf_manager.yaw_to_quaternion(math.pi)
        assert abs(quat_180['w']) < 0.01
        assert abs(abs(quat_180['z']) - 1.0) < 0.01

    def test_yaw_quaternion_roundtrip(self) -> None:
        """Test yaw-quaternion-yaw conversion roundtrip.

        Verifies that converting yaw to quaternion and back
        preserves the original value.
        """
        tf_manager = TransformManager()

        test_angles = [0.0, math.pi / 4, math.pi / 2, math.pi, -math.pi / 2]

        for original_yaw in test_angles:
            # Convert to quaternion and back
            quat = tf_manager.yaw_to_quaternion(original_yaw)
            # Note: _quaternion_to_yaw signature is (w, x, y, z)
            recovered_yaw = tf_manager._quaternion_to_yaw(
                quat['w'], quat['x'], quat['y'], quat['z']
            )

            # Account for angle wrapping
            diff = abs(original_yaw - recovered_yaw)
            if diff > math.pi:
                diff = 2 * math.pi - diff

            assert diff < 0.01, f'Roundtrip failed for {original_yaw}'


class TestTransformManagerDestroy:
    """Tests for TransformManager cleanup."""

    @patch('nav2_mcp_server.transforms.Node')
    def test_transform_manager_destroy(self, mock_node_class: Mock) -> None:
        """Test TransformManager cleanup.

        Verifies that the TransformManager properly cleans up
        ROS2 resources on destruction.
        """
        mock_node = Mock()
        mock_node_class.return_value = mock_node

        tf_manager = TransformManager()
        # Setup the node first
        tf_manager._node = mock_node

        # Now destroy
        tf_manager.destroy()

        # Verify node cleanup is called
        mock_node.destroy_node.assert_called_once()


class TestTransformManagerErrorHandling:
    """Tests for TransformManager error handling."""

    @patch('nav2_mcp_server.transforms.rclpy.spin_once')
    @patch('nav2_mcp_server.transforms.TransformListener')
    @patch('nav2_mcp_server.transforms.Buffer')
    @patch('nav2_mcp_server.transforms.Node')
    def test_get_robot_pose_timeout_error(
        self,
        mock_node_class: Mock,
        mock_buffer_class: Mock,
        mock_listener_class: Mock,
        mock_spin: Mock
    ) -> None:
        """Test robot pose retrieval with timeout error.

        Verifies that timeout errors are handled appropriately.
        """
        from tf2_ros import LookupException

        from nav2_mcp_server.exceptions import TransformError

        mock_node = Mock()
        mock_node_class.return_value = mock_node
        mock_buffer_instance = Mock()
        mock_buffer_class.return_value = mock_buffer_instance

        # Mock timeout exception
        mock_buffer_instance.can_transform.return_value = True
        mock_buffer_instance.lookup_transform.side_effect = LookupException(
            'Transform timeout'
        )

        tf_manager = TransformManager()
        context_manager = MCPContextManager()

        # Should raise TransformError instead of returning error dict
        with pytest.raises(TransformError):
            tf_manager.get_robot_pose(context_manager)

    def test_transform_manager_init_failure(self) -> None:
        """Test TransformManager initialization.

        Verifies that TransformManager uses lazy initialization
        and can be created without immediate ROS2 setup.
        """
        # Should not raise - node is created lazily when needed
        tf_manager = TransformManager()
        assert tf_manager is not None
        assert tf_manager._node is None  # Not initialized yet
