FROM ghcr.io/alpine-ros/alpine-ros:jazzy-3.20-ros-core

# ROS 2 environment variables
ENV ROS_DISTRO=jazzy
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV ROS_DOMAIN_ID=0
ENV ROS_LOCALHOST_ONLY=1

# Install necessary ROS2 packages
RUN apk add --no-cache \
    py3-pip \
    ros-${ROS_DISTRO}-rmw-fastrtps-cpp \
    ros-${ROS_DISTRO}-rosidl-generator-py \
    ros-${ROS_DISTRO}-rosidl-typesupport-c \
    ros-${ROS_DISTRO}-rosidl-typesupport-fastrtps-c \
    ros-${ROS_DISTRO}-rosidl-typesupport-fastrtps-cpp \
    ros-${ROS_DISTRO}-geometry-msgs \
    ros-${ROS_DISTRO}-lifecycle-msgs \
    ros-${ROS_DISTRO}-nav-msgs \
    ros-${ROS_DISTRO}-nav2-msgs \
    ros-${ROS_DISTRO}-nav2-simple-commander \
    && rm -rf /var/lib/apt/lists/*

# Install uv using pip
RUN pip3 install --no-cache-dir --break-system-packages uv

# Set working directory
WORKDIR /app

# Copy dependency files first for better Docker layer caching
COPY pyproject.toml .
COPY uv.lock* ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Copy source code
COPY src/ src/

# Install the package in development mode
RUN uv sync

# MCP server startup command using uv
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["uv", "run", "nav2_mcp_server"]