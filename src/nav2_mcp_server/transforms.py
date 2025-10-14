# Copyright (c) 2025 Alberto J. Tudela Roldán
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transform management for Nav2 MCP Server.

This module provides classes and functions for managing TF transforms
and robot pose operations.
"""

import math
from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformException, TransformListener

from .config import get_config
from .exceptions import NavigationErrorCode, TransformError
from .utils import MCPContextManager, safe_json_dumps


class TransformManager:
    """Manages TF transforms and robot pose operations."""

    def __init__(self) -> None:
        """Initialize the transform manager."""
        self.config = get_config()
        self._node: Optional[Node] = None
        self._tf_buffer: Optional[Buffer] = None
        self._tf_listener: Optional[TransformListener] = None

    def _ensure_tf_setup(self) -> None:
        """Ensure TF components are initialized."""
        if self._node is None:
            self._node = Node(self.config.logging.tf_node_name)
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self._node)

    def get_robot_pose(
        self, context_manager: MCPContextManager
    ) -> Dict[str, Any]:
        """Get current robot pose in map frame.

        Parameters
        ----------
        context_manager : MCPContextManager
            Context manager for logging.

        Returns
        -------
        dict
            Robot pose information including position and orientation.

        Raises
        ------
        TransformError
            If transform lookup fails.
        """
        context_manager.info_sync('Getting current robot pose...')

        self._ensure_tf_setup()

        # Ensure tf_buffer is available after setup
        if self._tf_buffer is None:
            raise TransformError(
                'TF buffer not initialized',
                NavigationErrorCode.TRANSFORM_UNAVAILABLE,
                {}
            )

        try:
            # Wait for transform to become available
            transform = None
            while transform is None:
                rclpy.spin_once(
                    self._node,
                    timeout_sec=self.config.navigation.default_tf_timeout
                )

                if self._tf_buffer.can_transform(
                    self.config.navigation.map_frame,
                    self.config.navigation.base_link_frame,
                    rclpy.time.Time()
                ):
                    transform = self._tf_buffer.lookup_transform(
                        self.config.navigation.map_frame,      # target frame
                        self.config.navigation.base_link_frame,  # source frame
                        rclpy.time.Time(),  # get latest available transform
                    )
                    break

            if transform is None:
                raise TransformError(
                    'Could not get transform after waiting',
                    NavigationErrorCode.TRANSFORM_UNAVAILABLE,
                    {
                        'target_frame': self.config.navigation.map_frame,
                        'source_frame': self.config.navigation.base_link_frame
                    }
                )

            # Extract pose information
            pose_info = self._extract_pose_from_transform(transform)

            context_manager.info_sync(
                f'Current robot pose: {safe_json_dumps(pose_info)}'
            )

            return pose_info

        except TransformException as ex:
            raise TransformError(
                f'Could not get transform from '
                f'{self.config.navigation.base_link_frame} '
                f'to {self.config.navigation.map_frame}: {str(ex)}',
                NavigationErrorCode.TRANSFORM_UNAVAILABLE,
                {
                    'target_frame': self.config.navigation.map_frame,
                    'source_frame': self.config.navigation.base_link_frame,
                    'tf_exception_type': type(ex).__name__,
                    'tf_exception_message': str(ex)
                }
            )
        except Exception as e:
            raise TransformError(
                f'Unexpected error getting robot pose: {str(e)}',
                NavigationErrorCode.TRANSFORM_UNAVAILABLE,
                {
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            )

    def _extract_pose_from_transform(self, transform: Any) -> Dict[str, Any]:
        """Extract pose information from transform.

        Parameters
        ----------
        transform
            TF transform message.

        Returns
        -------
        dict
            Pose information with position and orientation.
        """
        # Extract position
        position = {
            'x': transform.transform.translation.x,
            'y': transform.transform.translation.y,
            'z': transform.transform.translation.z
        }

        # Extract quaternion
        quat = {
            'w': transform.transform.rotation.w,
            'x': transform.transform.rotation.x,
            'y': transform.transform.rotation.y,
            'z': transform.transform.rotation.z
        }

        # Calculate yaw from quaternion (rotation around z-axis)
        yaw = self._quaternion_to_yaw(
            quat['w'], quat['x'], quat['y'], quat['z'])

        return {
            'position': position,
            'orientation': {
                'quaternion': quat,
                'yaw': yaw,
                'yaw_degrees': math.degrees(yaw)
            },
            'frame_id': self.config.navigation.map_frame,
            'child_frame_id': self.config.navigation.base_link_frame,
            'timestamp': {
                'sec': transform.header.stamp.sec,
                'nanosec': transform.header.stamp.nanosec
            }
        }

    def _quaternion_to_yaw(
        self, w: float, x: float, y: float, z: float
    ) -> float:
        """Convert quaternion to yaw angle.

        Parameters
        ----------
        w : float
            Quaternion w component.
        x : float
            Quaternion x component.
        y : float
            Quaternion y component.
        z : float
            Quaternion z component.

        Returns
        -------
        float
            Yaw angle in radians.
        """
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def yaw_to_quaternion(self, yaw: float) -> Dict[str, float]:
        """Convert yaw angle to quaternion.

        Parameters
        ----------
        yaw : float
            Yaw angle in radians.

        Returns
        -------
        dict
            Quaternion components (w, x, y, z).
        """
        return {
            'w': math.cos(yaw / 2.0),
            'x': 0.0,
            'y': 0.0,
            'z': math.sin(yaw / 2.0)
        }

    def destroy(self) -> None:
        """Clean up transform manager resources."""
        if self._tf_listener:
            # TF listener is cleaned up automatically with the node
            self._tf_listener = None

        if self._tf_buffer:
            self._tf_buffer = None

        if self._node:
            self._node.destroy_node()
            self._node = None


# Global transform manager instance
_transform_manager: Optional[TransformManager] = None


def get_transform_manager() -> TransformManager:
    """Get or create the global transform manager instance.

    Returns
    -------
    TransformManager
        The global transform manager instance.
    """
    global _transform_manager
    if _transform_manager is None:
        _transform_manager = TransformManager()
    return _transform_manager
