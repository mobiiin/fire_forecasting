"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    return dict(value) if isinstance(value, Mapping) else {}


def _normalize_sequence_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Keep legacy and nested sequence fields in sync after YAML loading."""

    for key in ("input_sequence_length", "prediction_horizon"):
        values: list[tuple[str, int]] = []
        if key in config:
            values.append((key, int(config[key])))
        for section_name in ("data", "sequence"):
            section = _section(config, section_name)
            if key in section:
                values.append((f"{section_name}.{key}", int(section[key])))
        if not values:
            continue
        unique_values = {value for _source, value in values}
        if len(unique_values) > 1:
            details = ", ".join(f"{source}={value}" for source, value in values)
            raise ValueError(f"Sequence config mismatch for {key}: {details}")
        config[key] = next(iter(unique_values))

    if "input_sequence_length" in config:
        input_sequence_length = int(config["input_sequence_length"])
        model = _section(config, "model")
        if model:
            model.setdefault("input_sequence_length", input_sequence_length)
            if int(model["input_sequence_length"]) != input_sequence_length:
                raise ValueError(
                    "Sequence config mismatch for input_sequence_length: "
                    f"input_sequence_length={input_sequence_length}, model.input_sequence_length={model['input_sequence_length']}"
                )
            config["model"] = model
        for section_name in ("convlstm_unet", "earthformer_lite", "st_mamba_lite", "cawfe_st_mamba", "weatherformer_lite", "cawfe_latte_lite", "cawfe_latte"):
            section = _section(config, section_name)
            if not section:
                continue
            section.setdefault("input_sequence_length", input_sequence_length)
            if int(section["input_sequence_length"]) != input_sequence_length:
                raise ValueError(
                    "Sequence config mismatch for input_sequence_length: "
                    f"input_sequence_length={input_sequence_length}, {section_name}.input_sequence_length={section['input_sequence_length']}"
                )
            config[section_name] = section

    for section_name in ("patching", "cache", "baselines"):
        section = _section(config, section_name)
        if not section:
            continue
        for key in ("input_sequence_length", "prediction_horizon"):
            if key not in config:
                continue
            section.setdefault(key, int(config[key]))
            if int(section[key]) != int(config[key]):
                raise ValueError(
                    f"Sequence config mismatch for {key}: {key}={config[key]}, {section_name}.{key}={section[key]}"
                )
        config[section_name] = section
    return config


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

    return _normalize_sequence_config(config)
