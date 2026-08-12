"""Run focused CAWFE-Latte hyperparameter tuning through the shared trainer."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

from src.config import load_config
from src.training.hyperparameter_tuning import (
	append_csv,
	append_jsonl,
	build_search_space,
	compute_composite_score,
	generate_trial_params,
	get_tuning_config,
	make_final_config_from_best_params,
	make_trial_config,
	save_json,
	save_yaml,
	select_best_history_row,
)
from src.training.train import _ensure_config_path, train_model_from_config


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Tune CAWFE-Latte hyperparameters with short shared-pipeline trials.")
	parser.add_argument("--config", default="configs/default.yaml", help="Base YAML config.")
	parser.add_argument("--num_trials", type=int, default=None, help="Number of tuning trials.")
	parser.add_argument("--output_dir", default=None, help="Output directory for tuning artifacts.")
	parser.add_argument("--method", choices=["random", "grid", "optuna"], default=None, help="Search method.")
	parser.add_argument("--seed", type=int, default=None, help="Random seed.")
	parser.add_argument("--resume", action="store_true", help="Skip completed trial result files.")
	parser.add_argument("--dry_run", action="store_true", help="Generate trial configs without training.")
	parser.add_argument("--trial_max_epochs", type=int, default=None, help="Epochs per trial.")
	parser.add_argument("--max_train_batches_per_epoch", type=int, default=None, help="Train-batch cap per trial.")
	parser.add_argument("--max_val_batches_per_epoch", type=int, default=None, help="Validation-batch cap per trial.")
	return parser


def _trial_summary_row(
	trial_id: int,
	params: dict[str, Any],
	status: str,
	best_score: float | None = None,
	best_row: dict[str, Any] | None = None,
	result: dict[str, Any] | None = None,
	runtime_seconds: float = 0.0,
	error_message: str = "",
) -> dict[str, Any]:
	best_row = best_row or {}
	result = result or {}
	return {
		"trial_id": trial_id,
		"status": status,
		"best_score": "" if best_score is None else best_score,
		"best_epoch": best_row.get("epoch", ""),
		"val_loss": best_row.get("val_loss", ""),
		"val_composite_score": best_row.get("val_composite_score", ""),
		"checkpoint_path": result.get("best_checkpoint_path", ""),
		"runtime_seconds": f"{float(runtime_seconds):.3f}",
		"error_message": error_message,
		"params_json": json.dumps(params, sort_keys=True),
	}


def _load_completed_result(path: Path) -> dict[str, Any] | None:
	if not path.exists():
		return None
	try:
		with path.open("r", encoding="utf-8") as handle:
			payload = json.load(handle)
	except json.JSONDecodeError:
		return None
	return payload if isinstance(payload, dict) and payload.get("status") == "completed" else None


def _save_summary(output_dir: Path, best_payload: dict[str, Any], trial_payloads: list[dict[str, Any]]) -> None:
	lines = [
		"CAWFE-Latte Hyperparameter Tuning Summary",
		"",
		f"Best trial: {best_payload['best_trial_id']}",
		f"Selection metric: {best_payload['selection_metric']} ({best_payload['selection_mode']})",
		f"Best score: {best_payload['best_score']}",
		"",
		"Best params:",
	]
	for key, value in best_payload["params"].items():
		lines.append(f"  {key}: {value}")
	lines.extend(["", f"Trials completed: {sum(1 for item in trial_payloads if item.get('status') == 'completed')}/{len(trial_payloads)}"])
	(output_dir / "tuning_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
	args = build_argument_parser().parse_args()
	base_config = _ensure_config_path(load_config(args.config), args.config)
	tuning_config = get_tuning_config(base_config)
	if str(tuning_config.get("architecture", "cawfe_latte")).lower() != "cawfe_latte":
		raise ValueError("This tuner is intentionally restricted to hparam_tuning.architecture=cawfe_latte.")

	num_trials = int(args.num_trials if args.num_trials is not None else tuning_config.get("num_trials", 12))
	output_dir = Path(args.output_dir or tuning_config.get("output_dir", "artifacts/hparam/cawfe_latte"))
	method = str(args.method or tuning_config.get("method", "random")).lower()
	if method == "optuna":
		try:
			import optuna  # noqa: F401
		except ImportError:
			print("Optuna is not installed; falling back to built-in random search.")
			method = "random"
		else:
			print("Optuna is available, but this no-new-dependencies runner uses the same sampled search space for now.")
			method = "random"
	seed = int(args.seed if args.seed is not None else tuning_config.get("seed", 42))
	selection_metric = str(tuning_config.get("selection_metric", "val_multitask_loss"))
	selection_mode = str(tuning_config.get("selection_mode", "min"))
	trial_max_epochs = int(args.trial_max_epochs if args.trial_max_epochs is not None else tuning_config.get("trial_max_epochs", 10))
	max_train_batches = args.max_train_batches_per_epoch
	if max_train_batches is None:
		max_train_batches = int(tuning_config.get("trial_overrides", {}).get("training", {}).get("performance", {}).get("max_train_batches_per_epoch", 1000))
	max_val_batches = args.max_val_batches_per_epoch
	if max_val_batches is None:
		max_val_batches = int(tuning_config.get("trial_overrides", {}).get("training", {}).get("validation", {}).get("max_val_batches_per_epoch", 200))

	output_dir.mkdir(parents=True, exist_ok=True)
	csv_path = output_dir / "tuning_trials.csv"
	jsonl_path = output_dir / "tuning_trials.jsonl"
	if not args.resume:
		csv_path.unlink(missing_ok=True)
		jsonl_path.unlink(missing_ok=True)

	trial_params = generate_trial_params(build_search_space(base_config), method=method, num_trials=num_trials, seed=seed)
	trial_payloads: list[dict[str, Any]] = []
	for trial_id, params in enumerate(trial_params):
		trial_dir = output_dir / f"trial_{trial_id:03d}"
		trial_dir.mkdir(parents=True, exist_ok=True)
		result_path = trial_dir / "result.json"
		if args.resume:
			completed = _load_completed_result(result_path)
			if completed is not None:
				print(f"trial_{trial_id:03d}: completed; skipping")
				trial_payloads.append(completed)
				continue

		trial_config = make_trial_config(
			base_config=base_config,
			params=params,
			trial_id=trial_id,
			output_dir=output_dir,
			trial_max_epochs=trial_max_epochs,
			max_train_batches_per_epoch=max_train_batches,
			max_val_batches_per_epoch=max_val_batches,
		)
		save_yaml(trial_dir / "config.yaml", trial_config)
		if args.dry_run:
			payload = {
				"trial_id": trial_id,
				"params": params,
				"status": "planned",
				"config_path": str(trial_dir / "config.yaml"),
			}
			trial_payloads.append(payload)
			continue

		start_time = time.perf_counter()
		status = "completed"
		error_message = ""
		result: dict[str, Any] = {}
		best_row = None
		best_score = None
		metric_used = selection_metric
		try:
			result = train_model_from_config(trial_config)
			history_rows = result.get("history_rows", [])
			best_row, best_score, metric_used = select_best_history_row(history_rows, selection_metric, selection_mode)
			if best_row is None or best_score is None:
				raise RuntimeError(f"No finite validation metric found for {selection_metric!r} or fallback val_multitask_loss.")
			if "val_composite_score" not in best_row:
				composite_score = compute_composite_score(best_row)
				if composite_score is not None:
					best_row["val_composite_score"] = composite_score
		except Exception as exc:  # pragma: no cover - exercised during real tuning failures
			status = "failed"
			error_message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
			(trial_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
		runtime_seconds = time.perf_counter() - start_time

		payload = {
			"trial_id": trial_id,
			"params": params,
			"status": status,
			"selection_metric": metric_used,
			"selection_mode": selection_mode,
			"best_score": best_score,
			"best_epoch": None if best_row is None else best_row.get("epoch"),
			"best_row": best_row,
			"checkpoint_path": result.get("best_checkpoint_path"),
			"training_log_path": result.get("training_log_path"),
			"runtime_seconds": runtime_seconds,
			"error_message": error_message,
		}
		save_json(result_path, payload)
		append_jsonl(jsonl_path, payload)
		append_csv(csv_path, _trial_summary_row(trial_id, params, status, best_score, best_row, result, runtime_seconds, error_message))
		trial_payloads.append(payload)
		print(f"trial_{trial_id:03d}: {status} score={best_score} runtime={runtime_seconds:.1f}s")

	if args.dry_run:
		save_json(output_dir / "planned_trials.json", {"trials": trial_payloads})
		print(f"Dry run wrote {len(trial_payloads)} trial configs under {output_dir}.")
		return

	completed_trials = [payload for payload in trial_payloads if payload.get("status") == "completed" and payload.get("best_score") is not None]
	if not completed_trials:
		raise RuntimeError("No completed tuning trials; cannot select best params.")
	reverse = selection_mode == "max"
	best_trial = sorted(completed_trials, key=lambda item: float(item["best_score"]), reverse=reverse)[0]
	best_payload = {
		"model_architecture": "cawfe_latte",
		"selection_metric": best_trial.get("selection_metric", selection_metric),
		"selection_mode": selection_mode,
		"best_trial_id": int(best_trial["trial_id"]),
		"best_score": float(best_trial["best_score"]),
		"params": best_trial["params"],
	}
	save_json(output_dir / "best_params.json", best_payload)
	best_config = make_final_config_from_best_params(base_config, best_payload, keep_trial_epochs=False)
	save_yaml(output_dir / "best_config.yaml", best_config)
	_save_summary(output_dir, best_payload, trial_payloads)
	print(f"Best trial: {best_payload['best_trial_id']} score={best_payload['best_score']}")
	print(f"Saved: {output_dir / 'best_params.json'}")
	print(f"Saved: {output_dir / 'best_config.yaml'}")


if __name__ == "__main__":
	main()
