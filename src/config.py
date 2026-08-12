"""Configuration loading helpers."""

from __future__ import annotations

import copy
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    return dict(value) if isinstance(value, Mapping) else {}


def compute_file_sha256(path: str | Path) -> str:
    """Return the SHA-256 hash for a file."""

    resolved = Path(path).expanduser().resolve()
    hasher = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_text_sha256(text: str) -> str:
    """Return the SHA-256 hash for UTF-8 text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key == "base_config":
            continue
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


_INTERPOLATION_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _lookup_dotted(config: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(f"Config interpolation references unknown key: {dotted_key!r}")
        current = current[part]
    return current


def _resolve_interpolations(value: Any, root: Mapping[str, Any]) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _resolve_interpolations(nested, root) for key, nested in value.items()}
    if isinstance(value, list):
        return [_resolve_interpolations(item, root) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_interpolations(item, root) for item in value)
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        replacement = _lookup_dotted(root, match.group(1).strip())
        return str(replacement)

    return _INTERPOLATION_PATTERN.sub(replace, value)


def resolve_config_interpolations(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve simple ${section.key} string references in a config mapping."""

    resolved = copy.deepcopy(dict(config))
    for _ in range(10):
        next_resolved = _resolve_interpolations(resolved, resolved)
        if next_resolved == resolved:
            return dict(next_resolved)
        resolved = next_resolved
    raise ValueError("Config interpolation did not converge after 10 passes.")


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


def _load_config(config_path: str | Path, *, finalize: bool) -> Dict[str, Any]:
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
        config = {}
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a mapping at the top level: {path}")

    base_config_value = config.get("base_config")
    base_path: Path | None = None
    if base_config_value not in (None, "", "null"):
        base_path = Path(str(base_config_value)).expanduser()
        if not base_path.is_absolute():
            base_path = (path.parent / base_path).resolve()
        base_config = _load_config(base_path, finalize=False)
        config = _deep_merge(base_config, config)
        config["base_config"] = str(base_path)

    if not finalize:
        return config

    config = resolve_config_interpolations(config)
    config["config_path"] = str(path)
    config["_config_path"] = str(path)
    config["_config_file_name"] = path.name
    config["_config_sha256"] = compute_file_sha256(path)
    if base_path is not None:
        config["_base_config_path"] = str(base_path)
        config["_base_config_sha256"] = compute_file_sha256(base_path)
    return _normalize_sequence_config(config)


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file into a fully resolved config dictionary."""

    return _load_config(config_path, finalize=True)
