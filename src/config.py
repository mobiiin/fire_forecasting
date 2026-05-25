"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file into a plain Python dictionary."""

    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in config file '{path}': {exc}") from exc

    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a mapping at the top level: {path}")

    return config
