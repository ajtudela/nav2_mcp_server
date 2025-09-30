"""Configuration settings for Nav2 MCP Server.

This module provides centralized configuration management with default values
and validation for the Nav2 MCP server application.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import logging


@dataclass
class NavigationConfig:
    """Configuration for navigation operations."""

    default_backup_speed: float = 0.2
    default_tf_timeout: float = 0.5
    localization_timeout: float = 1.0
    feedback_update_interval: int = 5

    # Frame names
    map_frame: str = 'map'
    base_link_frame: str = 'base_link'

    # Navigation limits
    max_waypoints: int = 100
    min_backup_distance: float = 0.01
    max_backup_distance: float = 10.0
    min_backup_speed: float = 0.01
    max_backup_speed: float = 1.0


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: int = logging.INFO
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    node_name: str = 'nav2_mcp_server'
    tf_node_name: str = 'nav2_mcp_server_tf_node'


@dataclass
class ServerConfig:
    """Main server configuration."""

    transport: str = 'stdio'
    server_name: str = 'nav2-mcp-server'
    description: str = 'MCP server wrapping Nav2 action clients'

    # Resource URIs
    status_uri: str = 'nav2://status'
    pose_uri: str = 'nav2://pose'


class Config:
    """Main configuration class that combines all config sections."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize configuration with optional override dictionary.

        Parameters
        ----------
        config_dict : dict, optional
            Dictionary to override default configuration values.
        """
        self.navigation = NavigationConfig()
        self.logging = LoggingConfig()
        self.server = ServerConfig()

        if config_dict:
            self._apply_overrides(config_dict)

        self._validate()

    def _apply_overrides(self, config_dict: Dict[str, Any]) -> None:
        """Apply configuration overrides from dictionary.

        Parameters
        ----------
        config_dict : dict
            Configuration overrides organized by section.
        """
        if 'navigation' in config_dict:
            nav_config = config_dict['navigation']
            for key, value in nav_config.items():
                if hasattr(self.navigation, key):
                    setattr(self.navigation, key, value)

        if 'logging' in config_dict:
            log_config = config_dict['logging']
            for key, value in log_config.items():
                if hasattr(self.logging, key):
                    setattr(self.logging, key, value)

        if 'server' in config_dict:
            server_config = config_dict['server']
            for key, value in server_config.items():
                if hasattr(self.server, key):
                    setattr(self.server, key, value)

    def _validate(self) -> None:
        """Validate configuration values.

        Raises
        ------
        ValueError
            If any configuration value is invalid.
        """
        nav = self.navigation

        if nav.default_backup_speed <= 0:
            raise ValueError("default_backup_speed must be positive")

        if nav.default_tf_timeout <= 0:
            raise ValueError("default_tf_timeout must be positive")

        if nav.max_waypoints <= 0:
            raise ValueError("max_waypoints must be positive")

        if nav.min_backup_distance >= nav.max_backup_distance:
            raise ValueError(
                "min_backup_distance must be less than max_backup_distance")

        if nav.min_backup_speed >= nav.max_backup_speed:
            raise ValueError(
                "min_backup_speed must be less than max_backup_speed")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns
        -------
        dict
            Configuration as nested dictionary.
        """
        return {
            'navigation': self.navigation.__dict__,
            'logging': self.logging.__dict__,
            'server': self.server.__dict__
        }


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance.

    Returns
    -------
    Config
        The global configuration instance.
    """
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance.

    Parameters
    ----------
    config : Config
        The configuration instance to set as global.
    """
    global _config
    _config = config


def load_config_from_dict(config_dict: Dict[str, Any]) -> Config:
    """Load configuration from dictionary and set as global.

    Parameters
    ----------
    config_dict : dict
        Configuration dictionary with override values.

    Returns
    -------
    Config
        The loaded configuration instance.
    """
    config = Config(config_dict)
    set_config(config)
    return config
