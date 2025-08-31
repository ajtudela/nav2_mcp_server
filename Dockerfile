FROM ros:jazzy-ros-core

# ROS 2 environment variables
ENV ROS_DISTRO=jazzy
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
SHELL ["/bin/bash", "-c"]

# Install necessary ROS2 packages (Nav2, geometry, etc.)
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-navigation2 \
    ros-${ROS_DISTRO}-nav2-bringup \
    ros-${ROS_DISTRO}-geometry-msgs \
    ros-${ROS_DISTRO}-nav-msgs \
    python3-pip python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY server.py .
COPY mcp.json .

# MCP server startup command
CMD ["python3", "server.py"]
