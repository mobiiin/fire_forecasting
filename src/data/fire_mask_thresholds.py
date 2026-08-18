"""Frozen fire-mask threshold resolution and criterion helpers."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping

THRESHOLD_KEYS=("energy_threshold_mw","surface_fuel_threshold","canopy_fuel_threshold")

def resolve_threshold_file(config: Mapping[str,Any], config_path: str|Path|None=None)->Path|None:
    section=config.get("target_construction",{}) if isinstance(config.get("target_construction"),Mapping) else {}; mask=section.get("fire_mask",{}) if isinstance(section.get("fire_mask"),Mapping) else {}; value=mask.get("threshold_file")
    if value in (None,"","null"): return None
    path=Path(str(value)).expanduser(); return path.resolve() if path.is_absolute() else ((Path(config_path).expanduser().resolve().parent if config_path else Path.cwd())/path).resolve()

def resolve_frozen_thresholds(config: Mapping[str,Any], config_path: str|Path|None=None, require: bool=True)->tuple[dict[str,float],dict[str,Any]]:
    section=config.get("target_construction",{}) if isinstance(config.get("target_construction"),Mapping) else {}; mask=section.get("fire_mask",{}) if isinstance(section.get("fire_mask"),Mapping) else {}; values={key:mask.get(key) for key in THRESHOLD_KEYS}
    if all(value is not None for value in values.values()): return {key:float(value) for key,value in values.items()},{"source":"config","threshold_file":None,"threshold_version":"config_frozen"}
    if any(value is not None for value in values.values()): raise ValueError("Fire-mask thresholds must specify all three values: energy, surface, and canopy.")
    path=resolve_threshold_file(config,config_path)
    if path is not None and path.exists():
        payload=json.loads(path.read_text(encoding="utf-8")); threshold=payload.get("thresholds",{}); missing=[key for key in THRESHOLD_KEYS if threshold.get(key) is None]
        if missing: raise ValueError(f"Threshold file {path} is missing: {missing}")
        return {key:float(threshold[key]) for key in THRESHOLD_KEYS},{"source":"file","threshold_file":str(path),"threshold_version":payload.get("threshold_version","unknown"),"payload":payload}
    if require: raise FileNotFoundError("Fire mask thresholds are missing. Run scripts/estimate_fire_mask_thresholds.py first.")
    return {},{"source":"missing","threshold_file":str(path) if path else None,"threshold_version":None}

def threshold_union_mask(energy_release_mw,surface_consumed,canopy_consumed,thresholds):
    return ((energy_release_mw>float(thresholds["energy_threshold_mw"]))|(surface_consumed>float(thresholds["surface_fuel_threshold"]))|(canopy_consumed>float(thresholds["canopy_fuel_threshold"]))).astype(bool)
