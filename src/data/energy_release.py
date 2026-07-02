"""Energy release map utilities derived from CAWFE flux channels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _get_section(config: Mapping[str, Any] | None, *names: str) -> dict[str, Any]:
    """Return the first nested mapping found under any of the provided names."""

    if not isinstance(config, Mapping):
        return {}
    for name in names:
        section = config.get(name)
        if isinstance(section, Mapping):
            return dict(section)
    return {}


def _project_root() -> Path:
    """Return the repository root from this module location."""

    return Path(__file__).resolve().parents[2]


def resolve_energy_release_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve energy release configuration with defaults."""

    section = _get_section(config, "energy_release")
    return {
        "enabled": bool(section.get("enabled", False)),
        "surface_sensible_flux_channel": int(section.get("surface_sensible_flux_channel", 80)),
        "surface_latent_flux_channel": int(section.get("surface_latent_flux_channel", 81)),
        "canopy_sensible_flux_channel": int(section.get("canopy_sensible_flux_channel", 82)),
        "canopy_latent_flux_channel": int(section.get("canopy_latent_flux_channel", 83)),
        "flux_units": str(section.get("flux_units", "W_per_m2")),
        "dx_m": section.get("dx_m"),
        "dy_m": section.get("dy_m"),
        "clamp_negative_flux_to_zero": bool(section.get("clamp_negative_flux_to_zero", True)),
        "save_components": bool(section.get("save_components", True)),
        "add_as_input_history": bool(section.get("add_as_input_history", False)),
        "target_transform": section.get("target_transform", "log1p"),
        "inverse_transform": section.get("inverse_transform", "expm1"),
        "predict_total": bool(section.get("predict_total", True)),
        "predict_sensible": bool(section.get("predict_sensible", False)),
        "predict_latent": bool(section.get("predict_latent", False)),
        "metadata_path": str(section.get("metadata_path", _project_root() / "fire_metadata.json")),
    }


def resolve_energy_output_channel_names(config: Mapping[str, Any]) -> list[str]:
    """Return the enabled energy release target names in channel order."""

    energy = resolve_energy_release_config(config)
    if not energy["enabled"]:
        return []

    channel_names: list[str] = []
    if energy["predict_total"]:
        channel_names.append("energy_release_total_MW")
    if energy["predict_sensible"]:
        channel_names.append("energy_release_sensible_MW")
    if energy["predict_latent"]:
        channel_names.append("energy_release_latent_MW")
    return channel_names


def resolve_energy_target_count(config: Mapping[str, Any]) -> int:
    """Return the number of enabled energy release targets."""

    return len(resolve_energy_output_channel_names(config))


def _resolve_metadata_path(config: Mapping[str, Any]) -> Path:
    """Resolve the fire metadata cache path."""

    energy = resolve_energy_release_config(config)
    metadata_path = Path(str(energy["metadata_path"])).expanduser()
    if metadata_path.is_absolute():
        return metadata_path.resolve()

    config_path_value = config.get("config_path", config.get("_config_path"))
    if config_path_value:
        config_path = Path(str(config_path_value)).expanduser().resolve()
        return (config_path.parent / metadata_path).resolve()
    return (_project_root() / metadata_path).resolve()


def load_fire_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    """Load the cached fire metadata JSON."""

    metadata_path = _resolve_metadata_path(config)
    if not metadata_path.exists():
        raise FileNotFoundError(
            "Energy release requires fire metadata with dx/dy, but the metadata cache was not found at "
            f"{metadata_path}. Run `python scripts/build_fire_metadata.py` for the relevant dataset root first."
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected fire metadata JSON to contain a top-level object, got {type(metadata)!r}.")
    return metadata


def _coerce_optional_float(value: Any) -> float | None:
    """Convert an optional numeric config value into float form."""

    if value in (None, "", "null"):
        return None
    return float(value)


def resolve_dataset_energy_geometry(
    data_dir: str | Path,
    config: Mapping[str, Any],
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """Resolve dx/dy/cell area for one dataset directory."""

    energy = resolve_energy_release_config(config)
    dx_m = _coerce_optional_float(energy["dx_m"])
    dy_m = _coerce_optional_float(energy["dy_m"])
    resolved_data_dir = Path(data_dir).expanduser().resolve()

    if dx_m is not None and dy_m is not None:
        cell_area_m2 = float(dx_m * dy_m)
        return {
            "dx_m": float(dx_m),
            "dy_m": float(dy_m),
            "cell_area_m2": cell_area_m2,
            "source": "config.energy_release.dx_m/dy_m",
            "data_dir": str(resolved_data_dir),
        }

    metadata = load_fire_metadata(config)
    data_dir_str = str(resolved_data_dir)

    for fire_name, record in metadata.items():
        if not isinstance(record, Mapping):
            continue
        simulations = record.get("simulations")
        if isinstance(simulations, list):
            for simulation in simulations:
                if not isinstance(simulation, Mapping):
                    continue
                simulation_dir = simulation.get("simulation_dir")
                if simulation_dir and str(Path(str(simulation_dir)).expanduser().resolve()) == data_dir_str:
                    dx_m = float(simulation["dx"])
                    dy_m = float(simulation["dy"])
                    return {
                        "dx_m": dx_m,
                        "dy_m": dy_m,
                        "cell_area_m2": float(dx_m * dy_m),
                        "source": f"fire_metadata.json simulation match ({fire_name})",
                        "fire_name": str(fire_name),
                        "data_dir": data_dir_str,
                    }

        fire_dir = record.get("fire_dir")
        if fire_dir and str(Path(str(fire_dir)).expanduser().resolve()) == data_dir_str:
            record_dx = record.get("dx")
            record_dy = record.get("dy")
            if record_dx is not None and record_dy is not None:
                dx_m = float(record_dx)
                dy_m = float(record_dy)
                return {
                    "dx_m": dx_m,
                    "dy_m": dy_m,
                    "cell_area_m2": float(dx_m * dy_m),
                    "source": f"fire_metadata.json fire match ({fire_name})",
                    "fire_name": str(fire_name),
                    "data_dir": data_dir_str,
                }

    if dataset_name and dataset_name in metadata:
        record = metadata[dataset_name]
        if isinstance(record, Mapping) and record.get("dx") is not None and record.get("dy") is not None:
            dx_m = float(record["dx"])
            dy_m = float(record["dy"])
            return {
                "dx_m": dx_m,
                "dy_m": dy_m,
                "cell_area_m2": float(dx_m * dy_m),
                "source": f"fire_metadata.json dataset_name match ({dataset_name})",
                "fire_name": str(dataset_name),
                "data_dir": data_dir_str,
            }

    raise KeyError(
        "Energy release requires dx/dy metadata for dataset "
        f"{resolved_data_dir}. No matching entry was found in {_resolve_metadata_path(config)}. "
        "Run `python scripts/build_fire_metadata.py` for this fire or for the full dataset root."
    )


def compute_energy_release_maps(
    frame: np.ndarray,
    config: Mapping[str, Any],
    dx_m: float | None = None,
    dy_m: float | None = None,
) -> dict[str, np.ndarray]:
    """Compute sensible/latent/total energy release maps in MW per grid cell."""

    energy = resolve_energy_release_config(config)
    raw_frame = np.asarray(frame, dtype=np.float32)
    if raw_frame.ndim != 3:
        raise ValueError(f"compute_energy_release_maps expects a frame shaped (H, W, C), got {raw_frame.shape}.")
    if str(energy["flux_units"]) != "W_per_m2":
        raise ValueError(
            "Unsupported energy_release.flux_units. "
            f"Expected 'W_per_m2', got {energy['flux_units']!r}."
        )

    surface_sensible = np.asarray(raw_frame[:, :, int(energy["surface_sensible_flux_channel"])], dtype=np.float32)
    surface_latent = np.asarray(raw_frame[:, :, int(energy["surface_latent_flux_channel"])], dtype=np.float32)
    canopy_sensible = np.asarray(raw_frame[:, :, int(energy["canopy_sensible_flux_channel"])], dtype=np.float32)
    canopy_latent = np.asarray(raw_frame[:, :, int(energy["canopy_latent_flux_channel"])], dtype=np.float32)

    if energy["clamp_negative_flux_to_zero"]:
        surface_sensible = np.maximum(surface_sensible, 0.0)
        surface_latent = np.maximum(surface_latent, 0.0)
        canopy_sensible = np.maximum(canopy_sensible, 0.0)
        canopy_latent = np.maximum(canopy_latent, 0.0)

    q_sensible_total = surface_sensible + canopy_sensible
    q_latent_total = surface_latent + canopy_latent
    q_total = q_sensible_total + q_latent_total

    resolved_dx = float(dx_m) if dx_m is not None else _coerce_optional_float(energy["dx_m"])
    resolved_dy = float(dy_m) if dy_m is not None else _coerce_optional_float(energy["dy_m"])
    if resolved_dx is None or resolved_dy is None:
        raise ValueError(
            "compute_energy_release_maps requires dx_m and dy_m either as arguments or in config.energy_release.dx_m/dy_m."
        )

    cell_area_m2 = float(resolved_dx * resolved_dy)
    energy_release_sensible_MW = (cell_area_m2 * q_sensible_total / 1.0e6).astype(np.float32, copy=False)
    energy_release_latent_MW = (cell_area_m2 * q_latent_total / 1.0e6).astype(np.float32, copy=False)
    energy_release_total_MW = (energy_release_sensible_MW + energy_release_latent_MW).astype(np.float32, copy=False)

    return {
        "energy_release_total_MW": energy_release_total_MW.astype(np.float32, copy=False),
        "energy_release_sensible_MW": energy_release_sensible_MW.astype(np.float32, copy=False),
        "energy_release_latent_MW": energy_release_latent_MW.astype(np.float32, copy=False),
        "q_sensible_total_W_m2": q_sensible_total.astype(np.float32, copy=False),
        "q_latent_total_W_m2": q_latent_total.astype(np.float32, copy=False),
        "q_total_W_m2": q_total.astype(np.float32, copy=False),
    }


def transform_energy_target(energy_MW: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    """Transform physical MW targets into the training space."""

    energy = resolve_energy_release_config(config)
    target_transform = energy["target_transform"]
    values = np.asarray(energy_MW, dtype=np.float32)
    if target_transform in ("log1p",):
        return np.log1p(np.maximum(values, 0.0)).astype(np.float32, copy=False)
    if target_transform in ("none", None, "null"):
        return values.astype(np.float32, copy=False)
    raise ValueError(
        "Unsupported energy_release.target_transform. "
        f"Expected 'log1p' or 'none', got {target_transform!r}."
    )


def inverse_transform_energy_target(y: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    """Invert the training-space energy target into physical MW."""

    energy = resolve_energy_release_config(config)
    target_transform = energy["target_transform"]
    inverse_transform = energy["inverse_transform"]
    values = np.asarray(y, dtype=np.float32)
    if inverse_transform == "expm1" or target_transform == "log1p":
        return np.expm1(values).astype(np.float32, copy=False)
    if target_transform in ("none", None, "null") or inverse_transform in ("none", None, "null"):
        return values.astype(np.float32, copy=False)
    raise ValueError(
        "Unsupported energy_release.inverse_transform. "
        f"Expected 'expm1' or 'none', got {inverse_transform!r}."
    )
