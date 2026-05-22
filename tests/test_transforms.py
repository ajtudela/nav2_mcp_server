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

    @patch('nav2_mcp_server.transforms.SingleThreadedExecutor')
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
        mock_spin: Mock,
        mock_executor_class: Mock
    ) -> None:
        """Test TransformManager TF setup.

        Verifies that TF2 buffer, listener, and dedicated spinner
        executor are properly configured.
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
        # Dedicated executor for the TF node is created and the node
        # is added to it (precondition for the background spinner).
        mock_executor_class.assert_called_once()
        mock_executor_class.return_value.add_node.assert_called_once_with(
            mock_node
        )

    @patch('nav2_mcp_server.transforms.threading.Thread')
    @patch('nav2_mcp_server.transforms.SingleThreadedExecutor')
    @patch('nav2_mcp_server.transforms.TransformListener')
    @patch('nav2_mcp_server.transforms.Buffer')
    @patch('nav2_mcp_server.transforms.Node')
    def test_ensure_tf_setup_starts_daemon_spinner_thread(
        self,
        mock_node_class: Mock,
        mock_buffer_class: Mock,
        mock_listener_class: Mock,
        mock_executor_class: Mock,
        mock_thread_class: Mock
    ) -> None:
        """Test that _ensure_tf_setup starts a daemon spinner thread.

        Verifies that the background TF spinner thread is created
        with daemon=True (so it does not block process shutdown)
        and is started exactly once.
        """
        mock_node_class.return_value = Mock()
        mock_buffer_class.return_value = Mock()
        mock_thread_instance = Mock()
        mock_thread_class.return_value = mock_thread_instance

        tf_manager = TransformManager()
        tf_manager._ensure_tf_setup()

        # Thread constructed exactly once with daemon=True.
        mock_thread_class.assert_called_once()
        kwargs = mock_thread_class.call_args.kwargs
        assert kwargs.get('daemon') is True
        # Spinner is named so it can be identified in stack traces.
        assert kwargs.get('name') == 'tf_spinner'
        # And the thread was actually started.
        mock_thread_instance.start.assert_called_once()

    @patch('nav2_mcp_server.transforms.threading.Thread')
    @patch('nav2_mcp_server.transforms.SingleThreadedExecutor')
    @patch('nav2_mcp_server.transforms.TransformListener')
    @patch('nav2_mcp_server.transforms.Buffer')
    @patch('nav2_mcp_server.transforms.Node')
    def test_ensure_tf_setup_is_idempotent(
        self,
        mock_node_class: Mock,
        mock_buffer_class: Mock,
        mock_listener_class: Mock,
        mock_executor_class: Mock,
        mock_thread_class: Mock
    ) -> None:
        """Test that _ensure_tf_setup is safe to call repeatedly.

        Verifies that resources are created on the first call only.
        Subsequent calls must not spawn additional nodes, executors,
        or spinner threads (which would otherwise duplicate TF
        callback processing).
        """
        mock_node_class.return_value = Mock()
        mock_buffer_class.return_value = Mock()
        mock_thread_class.return_value = Mock()

        tf_manager = TransformManager()
        tf_manager._ensure_tf_setup()
        tf_manager._ensure_tf_setup()
        tf_manager._ensure_tf_setup()

        # Resources created exactly once across multiple calls.
        mock_node_class.assert_called_once()
        mock_buffer_class.assert_called_once()
        mock_listener_class.assert_called_once()
        mock_executor_class.assert_called_once()
        mock_thread_class.assert_called_once()


class TestGetRobotPose:
    """Tests for robot pose retrieval functionality."""

    @patch('nav2_mcp_server.transforms.SingleThreadedExecutor')
    @patch('nav2_mcp_server.transforms.rclpy.spin_once')
    @patch('nav2_mcp_server.transforms.TransformListener')
    @patch('nav2_mcp_server.transforms.Buffer')
    @patch('nav2_mcp_server.transforms.Node')
    def test_get_robot_pose_success(
        self,
        mock_node_class: Mock,
        mock_buffer_class: Mock,
        mock_listener_class: Mock,
        mock_spin: Mock,
        mock_executor_class: Mock
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

    @patch('nav2_mcp_server.transforms.SingleThreadedExecutor')
    @patch('nav2_mcp_server.transforms.rclpy.spin_once')
    @patch('nav2_mcp_server.transforms.TransformListener')
    @patch('nav2_mcp_server.transforms.Buffer')
    @patch('nav2_mcp_server.transforms.Node')
    def test_get_robot_pose_transform_failure(
        self,
        mock_node_class: Mock,
        mock_buffer_class: Mock,
        mock_listener_class: Mock,
        mock_spin: Mock,
        mock_executor_class: Mock
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

    @patch('nav2_mcp_server.transforms.SingleThreadedExecutor')
    @patch('nav2_mcp_server.transforms.rclpy.spin_once')
    @patch('nav2_mcp_server.transforms.TransformListener')
    @patch('nav2_mcp_server.transforms.Buffer')
    @patch('nav2_mcp_server.transforms.Node')
    def test_get_robot_pose_with_custom_frames(
        self,
        mock_node_class: Mock,
        mock_buffer_class: Mock,
        mock_listener_class: Mock,
        mock_spin: Mock,
        mock_executor_class: Mock
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

    @patch('nav2_mcp_server.transforms.time.sleep')
    @patch('nav2_mcp_server.transforms.SingleThreadedExecutor')
    @patch('nav2_mcp_server.transforms.TransformListener')
    @patch('nav2_mcp_server.transforms.Buffer')
    @patch('nav2_mcp_server.transforms.Node')
    def test_get_robot_pose_waits_for_transform_availability(
        self,
        mock_node_class: Mock,
        mock_buffer_class: Mock,
        mock_listener_class: Mock,
        mock_executor_class: Mock,
        mock_sleep: Mock
    ) -> None:
        """Test that get_robot_pose polls until the transform is ready.

        The background spinner keeps the buffer current; the public
        method must poll can_transform and only call lookup_transform
        once the chain is available. Verifies the poll-then-lookup
        pattern by returning False for the first few polls and True
        thereafter.
        """
        mock_node_class.return_value = Mock()
        mock_buffer_instance = Mock()
        mock_buffer_class.return_value = mock_buffer_instance

        # Build a valid transform that will be returned on lookup.
        mock_transform = Mock()
        mock_transform.transform.translation.x = 1.0
        mock_transform.transform.translation.y = 2.0
        mock_transform.transform.translation.z = 0.0
        mock_transform.transform.rotation.x = 0.0
        mock_transform.transform.rotation.y = 0.0
        mock_transform.transform.rotation.z = 0.0
        mock_transform.transform.rotation.w = 1.0
        mock_transform.header.stamp.sec = 0
        mock_transform.header.stamp.nanosec = 0

        # Transform unavailable on first three polls, available on
        # the fourth. The method should sleep between polls and then
        # call lookup_transform exactly once.
        mock_buffer_instance.can_transform.side_effect = [
            False, False, False, True
        ]
        mock_buffer_instance.lookup_transform.return_value = mock_transform

        tf_manager = TransformManager()
        context_manager = MCPContextManager()

        result = tf_manager.get_robot_pose(context_manager)

        # Polled four times in total (three misses, one hit).
        assert mock_buffer_instance.can_transform.call_count == 4
        # Lookup happened exactly once, after the transform became
        # available — not on every iteration of the poll loop.
        mock_buffer_instance.lookup_transform.assert_called_once()
        # Slept between failed polls (at least three times).
        assert mock_sleep.call_count >= 3
        # Result reflects the transform that became available.
        assert result['position']['x'] == 1.0
        assert result['position']['y'] == 2.0


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

    @patch('nav2_mcp_server.transforms.Node')
    def test_transform_manager_destroy_with_tf_listener(
        self, mock_node_class: Mock
    ) -> None:
        """Test TransformManager cleanup with TF listener.

        Verifies that the TF listener is properly cleaned up.
        """
        mock_node = Mock()
        mock_node_class.return_value = mock_node
        mock_tf_listener = Mock()
        mock_tf_buffer = Mock()

        tf_manager = TransformManager()
        tf_manager._node = mock_node
        tf_manager._tf_listener = mock_tf_listener
        tf_manager._tf_buffer = mock_tf_buffer

        # Destroy
        tf_manager.destroy()

        # Verify cleanup
        assert tf_manager._tf_listener is None
        assert tf_manager._tf_buffer is None
        assert tf_manager._node is None
        mock_node.destroy_node.assert_called_once()


class TestTransformManagerGlobal:
    """Tests for global transform manager instance."""

    def test_get_transform_manager_creates_instance(self) -> None:
        """Test that get_transform_manager creates a new instance.

        Verifies that the global function creates a manager when needed.
        """
        from nav2_mcp_server import transforms

        # Reset global instance
        transforms._transform_manager = None

        # Get manager
        manager = transforms.get_transform_manager()

        assert manager is not None
        assert isinstance(manager, TransformManager)

    def test_get_transform_manager_returns_same_instance(self) -> None:
        """Test that get_transform_manager returns the same instance.

        Verifies that the global function returns a singleton instance.
        """
        from nav2_mcp_server import transforms

        # Reset global instance
        transforms._transform_manager = None

        # Get manager twice
        manager1 = transforms.get_transform_manager()
        manager2 = transforms.get_transform_manager()

        assert manager1 is manager2


class TestTransformManagerErrorHandling:
    """Tests for TransformManager error handling."""

    @patch('nav2_mcp_server.transforms.SingleThreadedExecutor')
    @patch('nav2_mcp_server.transforms.rclpy.spin_once')
    @patch('nav2_mcp_server.transforms.TransformListener')
    @patch('nav2_mcp_server.transforms.Buffer')
    @patch('nav2_mcp_server.transforms.Node')
    def test_get_robot_pose_timeout_error(
        self,
        mock_node_class: Mock,
        mock_buffer_class: Mock,
        mock_listener_class: Mock,
        mock_spin: Mock,
        mock_executor_class: Mock
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

    @patch('nav2_mcp_server.transforms.SingleThreadedExecutor')
    @patch('nav2_mcp_server.transforms.rclpy.spin_once')
    @patch('nav2_mcp_server.transforms.TransformListener')
    @patch('nav2_mcp_server.transforms.Buffer')
    @patch('nav2_mcp_server.transforms.Node')
    def test_get_robot_pose_buffer_not_initialized(
        self,
        mock_node_class: Mock,
        mock_buffer_class: Mock,
        mock_listener_class: Mock,
        mock_spin: Mock,
        mock_executor_class: Mock
    ) -> None:
        """Test robot pose retrieval when buffer is not initialized.

        Verifies error handling when TF buffer is unexpectedly None.
        """
        from nav2_mcp_server.exceptions import TransformError

        mock_node = Mock()
        mock_node_class.return_value = mock_node

        tf_manager = TransformManager()
        # Force tf_buffer to be None after setup
        tf_manager._ensure_tf_setup()
        tf_manager._tf_buffer = None

        context_manager = MCPContextManager()

        with pytest.raises(TransformError, match='TF buffer not initialized'):
            tf_manager.get_robot_pose(context_manager)

    @patch('nav2_mcp_server.transforms.time.sleep')
    @patch('nav2_mcp_server.transforms.SingleThreadedExecutor')
    @patch('nav2_mcp_server.transforms.rclpy.spin_once')
    @patch('nav2_mcp_server.transforms.TransformListener')
    @patch('nav2_mcp_server.transforms.Buffer')
    @patch('nav2_mcp_server.transforms.Node')
    def test_get_robot_pose_transform_timeout(
        self,
        mock_node_class: Mock,
        mock_buffer_class: Mock,
        mock_listener_class: Mock,
        mock_spin: Mock,
        mock_executor_class: Mock,
        mock_sleep: Mock
    ) -> None:
        """Test robot pose when transform never becomes available.

        Verifies that the wait-until-available polling loop exits
        and raises TransformError when can_transform never succeeds.
        """
        from nav2_mcp_server.exceptions import TransformError

        mock_node = Mock()
        mock_node_class.return_value = mock_node
        mock_buffer_instance = Mock()
        mock_buffer_class.return_value = mock_buffer_instance

        # Transform never becomes available throughout the polling
        # window. The 0.1 s sleep between polls is mocked out so the
        # loop completes its 50 iterations quickly.
        mock_buffer_instance.can_transform.return_value = False

        tf_manager = TransformManager()
        context_manager = MCPContextManager()

        with pytest.raises(TransformError, match='after waiting'):
            tf_manager.get_robot_pose(context_manager)

        # Polling loop ran the full budget and slept between attempts.
        assert mock_sleep.called
        assert mock_buffer_instance.can_transform.call_count >= 10
