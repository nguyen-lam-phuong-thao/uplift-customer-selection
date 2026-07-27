"""Shared configuration and project-path helpers."""

from pathlib import Path
from typing import Any

import yaml


def get_project_root(anchor_path: Path) -> Path:
    """Find the project root from a source file path."""
    resolved_anchor = anchor_path.resolve()

    for parent in resolved_anchor.parents:
        if (parent / "src" / "uplift_modeling").exists():
            return parent

    raise ValueError(f"Could not resolve project root from: {anchor_path}")


def resolve_project_path(path_value: str | Path, project_root: Path) -> Path:
    """Resolve an absolute or project-relative path."""
    path = Path(path_value).expanduser()

    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path.resolve()

    return (project_root / path).resolve()


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML config file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    return config


def get_config_section(
    config: dict[str, Any],
    section_name: str,
) -> dict[str, Any]:
    """Return a required top-level config section."""
    section = config.get(section_name)

    if not isinstance(section, dict):
        raise ValueError(
            f"Config section '{section_name}' must be present as a mapping."
        )

    return section
