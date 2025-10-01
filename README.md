# Nav2 MCP Server

[![License](https://img.shields.io/github/license/ajtudela/nav2_mcp_server)](https://github.com/ajtudela/nav2_mcp_server/blob/main/LICENSE)

An MCP (Model Context Protocol) server that provides tools and resources to control and monitor Nav2 navigation operations, allowing seamless integration with Nav2-enabled robots through the MCP protocol.

## Tools

| Tool                      | Description                                                                       | Parameters                                                                        |
| ------------------------- | --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **navigate_to_pose**      | Navigate the robot to a specific pose (position and orientation) in the map frame | `x: float, y: float, yaw: float`                                                  |
| **follow_waypoints**      | Navigate the robot through a sequence of waypoints in order                       | `waypoints: str (JSON array)`                                                     |
| **spin_robot**            | Rotate the robot in place by a specified angle                                    | `angle: float`                                                                    |
| **backup_robot**          | Move the robot backward by a specified distance                                   | `distance: float, speed: float`                                                   |
| **dock_robot**            | Dock the robot to a charging station or dock                                      | `x: float, y: float, yaw: float, dock_id: str, dock_type: str, nav_to_dock: bool` |
| **undock_robot**          | Undock the robot from a charging station or dock                                  | `dock_type: str`                                                                  |
| **clear_costmaps**        | Clear robot navigation costmaps to remove stale obstacle data                     | `costmap_type: str`                                                               |
| **get_robot_pose**        | Get the current position and orientation of the robot                             | —                                                                                 |
| **get_navigation_status** | Get the current status of the navigation system and any active tasks              | —                                                                                 |
| **cancel_navigation**     | Cancel the currently active navigation task                                       | —                                                                                 |
| **nav2_lifecycle**        | Control Nav2 lifecycle (startup or shutdown)                                      | `action: str`                                                                     |

## Environment Variables

| Variable             | Default | Description                                                  |
| -------------------- | ------- | ------------------------------------------------------------ |
| `ROS_DOMAIN_ID`      | —       | ROS 2 domain ID for network isolation (recommended to set)   |
| `ROS_LOCALHOST_ONLY` | —       | Set to '1' to restrict ROS 2 communication to localhost only |

## Features

* **Navigation control**: Navigate to specific poses, follow waypoint sequences, and execute precise robot movements
* **Real-time status**: Monitor navigation progress, robot pose, and system status with comprehensive feedback
* **Costmap management**: Clear stale obstacle data and manage navigation costmaps for optimal path planning
* **Lifecycle management**: Control Nav2 system startup and shutdown for complete system control
* **ROS 2 integration**: Full compatibility with Nav2 navigation stack and ROS 2 ecosystem
* **Async operations**: Non-blocking navigation commands with progress monitoring and cancellation support

## Installation

### Dependencies

- [Robot Operating System (ROS) 2](https://docs.ros.org/en/jazzy/): Middleware for robotics (Jazzy)
- [fastmcp](https://github.com/jlowin/fastmcp): MCP server framework
- [python](https://www.python.org/): Python programming language
- [uv](https://github.com/astral-sh/uv): Python package manager (optional)


### Install with uv (recommended)

Clone the repository and install with uv:

```bash
git clone https://github.com/ajtudela/nav2_mcp_server.git
cd nav2_mcp_server
# Set up ROS 2 environment variables if needed
export ROS_DOMAIN_ID=0
uv sync
```

Or install directly from the repository:

```bash
uv add git+https://github.com/ajtudela/nav2_mcp_server.git
```

### Install with pip

Install the package in development mode:

```bash
git clone https://github.com/ajtudela/nav2_mcp_server.git
cd nav2_mcp_server
# Set up ROS 2 environment variables if needed
export ROS_DOMAIN_ID=0
python3 -m pip install -e .
```

Or install directly from the repository:

```bash
python3 -m pip install git+https://github.com/ajtudela/nav2_mcp_server.git
```

### Docker

#### Build the image
Build the image:

```bash
docker build -t nav2_mcp_server:latest .
```

#### Pull the image
Pull the latest image from the Docker registry:

```bash
docker pull ghcr.io/ajtudela/nav2_mcp_server:latest
```

## Usage

### Running with uv

```bash
uv run nav2_mcp_server
```

### Running with pip installation

```bash
python3 -m nav2_mcp_server
```

### Configuration example for Claude Desktop/Cursor/VSCode

Add this configuration to your application's settings (mcp.json):

#### Using uv (recommended)
```json
{
  "nav2 mcp server": {
    "type": "stdio",
    "command": "uv",
    "args": [
      "run",
      "--directory",
      "/path/to/nav2_mcp_server",
      "nav2_mcp_server"
    ],
    "env": {
      "ROS_DOMAIN_ID": "0",
      "ROS_LOCALHOST_ONLY": "1"
    }
  }
}
```

#### Using pip installation
```json
{
  "nav2 mcp server": {
    "type": "stdio",
    "command": "python3",
    "args": [
      "-m",
      "nav2_mcp_server"
    ],
    "env": {
      "ROS_DOMAIN_ID": "0",
      "ROS_LOCALHOST_ONLY": "1"
    }
  }
}
```

#### Using Docker
```json
"nav2 mcp server": {
    "type": "stdio",
    "command": "docker",
    "args": [
        "run",
        "-i",
        "--rm",
        "ghcr.io/ajtudela/nav2_mcp_server"
    ],
    "env": {
      "ROS_DOMAIN_ID": "0",
      "ROS_LOCALHOST_ONLY": "1"
    }
}
```