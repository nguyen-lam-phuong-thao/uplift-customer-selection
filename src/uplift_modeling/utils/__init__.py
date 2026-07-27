"""Small shared utilities for configuration, paths, logging, and validation."""

from uplift_modeling.utils.config import (
    get_config_section,
    get_project_root,
    load_yaml_config,
    resolve_project_path,
)

__all__ = [
    "get_config_section",
    "get_project_root",
    "load_yaml_config",
    "resolve_project_path",
]
