"""Resolve the frozen Table 2 runs used by FLARE paper utilities."""

from __future__ import annotations

import json
from pathlib import Path

from src.config import load_config
from src.training.input_normalization import resolve_input_normalization_stats_path

ROOT = Path(__file__).resolve().parents[2]
PAPER_NAMES = {
    "persistence": "Persistence",
    "cawfe_st_mamba": "CAWFE-ST-Mamba",
    "cawfe_latte_baseline": "FLARE baseline",
    "cawfe_latte_final": "FLARE final",
}
REPO_NAMES = {
    "persistence": "persistence",
    "cawfe_st_mamba": "st_mamba_lite",
    "cawfe_latte_baseline": "cawfe_latte",
    "cawfe_latte_final": "cawfe_latte",
}
TEST_FIRES = (
    "CHIMNEYTOPS2", "MCKINNEY", "RIM__0822__keepz_08", "SPRINGS",
    "THOMAS__1210__keepz_08", "TUOLUMNE_65", "WOOLSEY",
)


def _parent(alias: str, seed: int | None) -> Path:
    if alias == "persistence":
        return ROOT / "artifacts/table2_baselines/persistence"
    if seed is None:
        raise ValueError(f"A checkpoint seed is required for {alias}")
    if alias == "cawfe_st_mamba":
        return ROOT / "artifacts/table2_baselines/cawfe_st_mamba" / f"seed_{seed}"
    finalist = "baseline" if alias == "cawfe_latte_baseline" else "GA_Q2"
    return ROOT / "artifacts/final_training/cawfe_latte" / finalist / f"seed_{seed}"


def resolve_run(alias: str, seed: int | None = None, *, require_metrics: bool = False) -> dict:
    if alias not in PAPER_NAMES:
        raise ValueError(f"Unknown model {alias!r}; choose from {', '.join(PAPER_NAMES)}")
    parent = _parent(alias, seed)
    summary_path = ROOT / "artifacts/table2_baselines/summary/table2_results.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = [row for row in summary["rows"] if row.get("model_key") == alias]
    if len(rows) != 1:
        raise ValueError(f"Table 2 summary does not identify exactly one row for {alias}: {summary_path}")
    row = rows[0]
    matches = [ROOT / directory for found_seed, directory in zip(row["seed_values"], row["run_dirs"])
               if found_seed == seed]
    if len(matches) != 1:
        raise FileNotFoundError(f"Table 2 has no selected run for {alias} seed={seed}")
    run = matches[0]
    if not run.is_dir() or run != parent and parent not in run.parents:
        raise ValueError(f"Selected Table 2 run is outside the expected model/seed directory: {run}")
    if require_metrics and not ((run / "evaluation/test_metrics.json").is_file()
                                and (run / "evaluation/test_per_fire_metrics.csv").is_file()):
        raise FileNotFoundError(f"Selected Table 2 run lacks complete per-fire metrics: {run}")
    config_path = run / "resolved_config.yaml"
    if not config_path.is_file():
        config_path = run / "configs/resolved_config.yaml"
    config = load_config(config_path)
    if alias != "persistence":
        architecture = str(config.get("model", {}).get("architecture"))
        if architecture != REPO_NAMES[alias]:
            raise ValueError(f"Architecture mismatch at {config_path}: {architecture}")
        checkpoint = run / "checkpoints/best_model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    else:
        checkpoint = None
    normalization = None if alias == "persistence" else resolve_input_normalization_stats_path(config, must_exist=True)
    return {"alias": alias, "seed": seed, "run_dir": run, "config_path": config_path,
            "config": config, "checkpoint": checkpoint, "normalization": normalization,
            "paper_model_name": PAPER_NAMES[alias], "repo_model_name": REPO_NAMES[alias]}


def prepare_cpu_mamba(model) -> None:
    """Use mamba_ssm's own reference scan for frozen CPU inference.

    The installed fused causal convolution and selective scan kernels require
    CUDA. This keeps the trained Mamba parameters and exact recurrence while
    using the package's PyTorch reference operations on CPU.
    """
    from src.models.mamba_backend import MambaLayerWrapper
    import mamba_ssm.modules.mamba_simple as mamba_simple
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
    mamba_simple.causal_conv1d_fn = None
    mamba_simple.selective_scan_fn = selective_scan_ref
    for module in model.modules():
        if isinstance(module, MambaLayerWrapper):
            module.layer.use_fast_path = False
