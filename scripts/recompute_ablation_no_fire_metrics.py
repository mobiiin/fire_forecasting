#!/usr/bin/env python3
"""Recompute target-defined no-fire metrics on complete CAWFE-Latte validation splits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from scripts.check_patch_fire_balance import check_balance
from src.config import load_config
from src.data.dataset import create_dataloaders
from src.evaluation.no_fire_metrics import FullValidationNoFireAccumulator
from src.models.model_factory import build_model_from_config
from src.training.batch_utils import unpack_batch
from src.training.checkpoints import load_checkpoint, load_model_state_dict_compatible, validate_checkpoint_model_compatibility
from src.training.hardware import autocast_context, choose_amp_dtype
from src.training.input_normalization import apply_input_normalization, build_input_normalizer_for_loader
from src.training.model_outputs import extract_prediction


DEFAULT_ROOT = PROJECT_ROOT / "artifacts" / "ablations" / "cawfe_latte"
DEFAULT_SLURM_SCRIPT = PROJECT_ROOT / "scripts" / "slurm_recompute_cawfe_latte_no_fire_metrics_a10080.sh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true", help="Evaluate all ablations (all completed runs locally; latest run per ablation with --submit-slurm).")
    selection.add_argument("--ablation", action="append", help="Evaluate one named ablation; repeat for multiple names.")
    selection.add_argument("--run-dir", action="append", help="Evaluate one exact completed run directory; repeat for multiple runs.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--submit-slurm",
        action="store_true",
        help="Submit one independent Slurm job per selected ablation instead of evaluating in this process.",
    )
    parser.add_argument("--slurm-script", default=str(DEFAULT_SLURM_SCRIPT), help=argparse.SUPPRESS)
    return parser.parse_args()


def apply_loader_overrides(config: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = dict(config)
    updated["return_metadata"] = False
    training = dict(updated.get("training", {}))
    data_loader = dict(updated.get("data_loader", {}))
    if args.batch_size is not None:
        if int(args.batch_size) <= 0:
            raise ValueError("--batch-size must be positive.")
        updated["batch_size"] = int(args.batch_size)
        training["batch_size"] = int(args.batch_size)
        data_loader["batch_size"] = int(args.batch_size)
    if int(args.num_workers) < 0:
        raise ValueError("--num-workers must be nonnegative.")
    training["num_workers"] = int(args.num_workers)
    training["persistent_workers"] = False
    data_loader["num_workers"] = int(args.num_workers)
    data_loader["persistent_workers"] = False
    updated["training"] = training
    updated["data_loader"] = data_loader
    return updated


def resolve_device(value: str) -> torch.device:
    text = str(value).lower()
    if text == "auto":
        text = "cuda" if torch.cuda.is_available() else "cpu"
    if text == "gpu":
        text = "cuda"
    if text.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"--device {value} was requested, but CUDA is unavailable.")
    return torch.device(text)


def run_ablation_name(run_dir: Path) -> str:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        return str(payload.get("ablation", run_dir.parent.name))
    return run_dir.parent.name


def discover_runs(root: Path, requested: set[str] | None) -> list[Path]:
    runs: list[Path] = []
    for metrics_path in sorted(root.glob("*/*/metrics.json")):
        run_dir = metrics_path.parent
        if requested and run_ablation_name(run_dir) not in requested and run_dir.parent.name not in requested:
            continue
        config_path = run_dir / "resolved_config.yaml"
        if not config_path.is_file():
            config_path = run_dir / "configs" / "resolved_config.yaml"
        checkpoint_path = run_dir / "checkpoints" / "best_model.pt"
        if config_path.is_file() and checkpoint_path.is_file():
            runs.append(run_dir)
    if requested:
        found = {run_ablation_name(run) for run in runs} | {run.parent.name for run in runs}
        missing = requested - found
        if missing:
            raise ValueError(f"No completed run found for ablation(s): {sorted(missing)}")
    return runs


def validate_explicit_run(run_dir: str | Path) -> Path:
    """Validate and resolve one exact completed run directory."""
    resolved = Path(run_dir).expanduser().resolve()
    metrics_path = resolved / "metrics.json"
    checkpoint_path = resolved / "checkpoints" / "best_model.pt"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Completed run metrics are missing: {metrics_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Best checkpoint is missing: {checkpoint_path}")
    resolve_config_path(resolved)
    return resolved


def latest_runs_by_ablation(runs: list[Path]) -> list[Path]:
    """Select the newest completed run for each ablation deterministically."""
    latest: dict[str, Path] = {}
    for run_dir in runs:
        name = run_ablation_name(run_dir)
        previous = latest.get(name)
        if previous is None:
            latest[name] = run_dir
            continue
        current_key = ((run_dir / "metrics.json").stat().st_mtime, str(run_dir))
        previous_key = ((previous / "metrics.json").stat().st_mtime, str(previous))
        if current_key > previous_key:
            latest[name] = run_dir
    return [latest[name] for name in sorted(latest)]


def pending_submission_runs(runs: list[Path], *, overwrite: bool) -> list[Path]:
    """Avoid allocating Slurm jobs for runs whose sidecar is already complete."""
    if overwrite:
        return list(runs)
    return [run_dir for run_dir in runs if not (run_dir / "no_fire_metrics.json").is_file()]


def build_sbatch_command(args: argparse.Namespace, run_dir: Path, root: Path) -> list[str]:
    """Build the non-shell Slurm command for one exact run."""
    return [
        "sbatch",
        "--parsable",
        str(Path(args.slurm_script).expanduser().resolve()),
        str(run_dir.resolve()),
        str(root.resolve()),
        str(args.device),
        "default" if args.batch_size is None else str(int(args.batch_size)),
        str(int(args.num_workers)),
        "1" if bool(args.overwrite) else "0",
    ]


def submit_slurm_jobs(args: argparse.Namespace, runs: list[Path], root: Path) -> list[str]:
    """Submit one independent full-validation job per selected ablation run."""
    script_path = Path(args.slurm_script).expanduser().resolve()
    if not script_path.is_file():
        raise FileNotFoundError(f"No-fire Slurm worker script not found: {script_path}")
    (PROJECT_ROOT / "artifacts" / "logs" / "slurm").mkdir(parents=True, exist_ok=True)
    job_ids: list[str] = []
    for run_dir in runs:
        command = build_sbatch_command(args, run_dir, root)
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
        job_id = completed.stdout.strip()
        if not job_id:
            raise RuntimeError(f"sbatch returned an empty job ID for {run_dir}")
        job_ids.append(job_id)
        print(f"{run_ablation_name(run_dir)}: submitted no-fire job {job_id} for {run_dir}")
    return job_ids


def resolve_config_path(run_dir: Path) -> Path:
    candidates = (run_dir / "resolved_config.yaml", run_dir / "configs" / "resolved_config.yaml")
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No resolved config found under {run_dir}")


def dataset_identity(config: Mapping[str, Any]) -> tuple[Path, str]:
    dataloader = config.get("dataloader", {})
    processed = config.get("processed_dataset", {})
    if not isinstance(dataloader, Mapping) or str(dataloader.get("source", "")).lower() != "processed_full_frames":
        raise ValueError("Post-hoc no-fire evaluation requires dataloader.source=processed_full_frames.")
    root_value = dataloader.get("dataset_root", processed.get("root") if isinstance(processed, Mapping) else None)
    if root_value is None:
        raise KeyError("Processed dataset root is missing from the resolved config.")
    return Path(str(root_value)).expanduser().resolve(), str(dataloader.get("sample_pattern", "consecutive5_h10"))


def call_model(model: torch.nn.Module, x: torch.Tensor, terrain: torch.Tensor | None):
    if terrain is None:
        return model(x)
    return model(x, terrain=terrain)


def evaluate_run(
    run_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    balance_cache: dict[tuple[str, str], dict[str, Any]],
) -> Path:
    output_path = run_dir / "no_fire_metrics.json"
    if output_path.exists() and not args.overwrite:
        print(f"SKIP {run_ablation_name(run_dir)}: {output_path} already exists")
        return output_path

    config_path = resolve_config_path(run_dir)
    config = apply_loader_overrides(load_config(config_path), args)
    dataset_root, pattern = dataset_identity(config)
    balance_key = (str(dataset_root), pattern)
    if balance_key not in balance_cache:
        print(f"Canonical full-val balance scan: root={dataset_root} pattern={pattern}", flush=True)
        balance_cache[balance_key] = check_balance(
            dataset_root,
            sample_pattern=pattern,
            split="val",
            fire_threshold=0.5,
            active_fraction_threshold=0.0,
            verbose=True,
        )
    canonical_counts = dict(balance_cache[balance_key]["counts"])

    _train_loader, val_loader, _test_loader = create_dataloaders(config)
    try:
        first_batch = next(iter(val_loader))
    except StopIteration as exc:
        raise ValueError(f"Validation loader is empty for {run_dir}") from exc
    first_x, _first_y, _first_extra = unpack_batch(first_batch)
    if first_x.ndim != 5:
        raise ValueError(f"Expected validation inputs shaped (B, T, C, H, W), got {tuple(first_x.shape)}")
    input_channels = int(first_x.shape[2])

    checkpoint_path = run_dir / "checkpoints" / "best_model.pt"
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model = build_model_from_config(config, input_channels=input_channels).to(device)
    validate_checkpoint_model_compatibility(model, checkpoint, checkpoint_path)
    load_model_state_dict_compatible(model, checkpoint, checkpoint_path)
    model.eval()
    amp_dtype = choose_amp_dtype(config, device)
    normalizer = build_input_normalizer_for_loader(val_loader, device, input_channels)
    accumulator = FullValidationNoFireAccumulator()

    print(
        f"EVAL {run_ablation_name(run_dir)} | patches={len(val_loader.dataset)} "
        f"batches={len(val_loader)} device={device} checkpoint={checkpoint_path}",
        flush=True,
    )
    with torch.inference_mode():
        for batch_index, batch in enumerate(val_loader, start=1):
            x_raw, y_raw, extra = unpack_batch(batch)
            terrain_raw = extra.get("terrain")
            x = x_raw.to(device, non_blocking=True)
            y = y_raw.to(device, non_blocking=True).float()
            terrain = terrain_raw.to(device, non_blocking=True) if terrain_raw is not None else None
            x = apply_input_normalization(x, normalizer)
            with autocast_context(device, amp_dtype):
                output = call_model(model, x, terrain)
            prediction = extract_prediction(output).float()
            accumulator.update(prediction, y)
            if batch_index == 1 or batch_index % 100 == 0 or batch_index == len(val_loader):
                print(f"  {batch_index}/{len(val_loader)} batches", flush=True)

    metrics = accumulator.finalize()
    expected_triplet = (
        int(canonical_counts["total"]),
        int(canonical_counts["fire"]),
        int(canonical_counts["no_fire"]),
    )
    evaluated_triplet = (
        int(metrics["full_val_total_patch_count"]),
        int(metrics["full_val_fire_patch_count"]),
        int(metrics["full_val_no_fire_patch_count"]),
    )
    if int(canonical_counts["no_fire"]) > 0 and int(metrics["full_val_no_fire_patch_count"]) == 0:
        raise RuntimeError("Canonical balance checker found no-fire validation patches, but post-hoc evaluation counted zero.")
    if evaluated_triplet != expected_triplet:
        raise RuntimeError(f"Post-hoc validation counts {evaluated_triplet} disagree with canonical counts {expected_triplet}.")

    payload = {
        "schema_version": 1,
        "ablation": run_ablation_name(run_dir),
        "run_directory": str(run_dir.resolve()),
        "resolved_config": str(config_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metric_scope": "full_validation_posthoc",
        "split": "val",
        "classification": {
            "source": "ground_truth_target_mask_only",
            "mask_channel": 2,
            "fire_pixel_rule": "target_mask > 0.5",
            "fire_patch_rule": "at least one fire pixel",
            "no_fire_patch_rule": "zero fire pixels",
            "prediction_threshold": 0.5,
        },
        "canonical_balance_counts": canonical_counts,
        "metrics": metrics,
    }
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)
    print(
        "FULL VAL "
        f"total={metrics['full_val_total_patch_count']} "
        f"fire={metrics['full_val_fire_patch_count']} "
        f"no_fire={metrics['full_val_no_fire_patch_count']} "
        f"no_fire_percent={metrics['full_val_no_fire_percent']:.6f}",
        flush=True,
    )
    print(f"WROTE {output_path}", flush=True)
    return output_path


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if args.run_dir:
        runs = [validate_explicit_run(value) for value in args.run_dir]
    else:
        requested = set(args.ablation or []) or None
        runs = discover_runs(root, requested)
    if not runs:
        raise ValueError(f"No completed CAWFE-Latte runs found under {root}")

    if args.submit_slurm:
        # Match the training batch launcher: one job per ablation. When an
        # ablation has been rerun, target only its newest completed run.
        submission_runs = runs if args.run_dir else latest_runs_by_ablation(runs)
        submission_runs = pending_submission_runs(submission_runs, overwrite=bool(args.overwrite))
        if not submission_runs:
            print("All selected latest runs already contain no_fire_metrics.json; no Slurm jobs submitted.")
            return
        job_ids = submit_slurm_jobs(args, submission_runs, root)
        print(f"Submitted {len(job_ids)} no-fire Slurm job(s).")
        print("After the jobs finish, run: python scripts/summarize_cawfe_latte_ablations.py")
        return

    device = resolve_device(args.device)
    balance_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for run_dir in runs:
        evaluate_run(run_dir, args, device, balance_cache)
    print(f"Completed post-hoc no-fire evaluation for {len(runs)} run(s).")


if __name__ == "__main__":
    main()
