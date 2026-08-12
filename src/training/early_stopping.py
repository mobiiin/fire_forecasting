"""Shared early-stopping helper for training loops."""

from __future__ import annotations

import math
from typing import Any, Mapping


class EarlyStopping:
	"""Track validation-metric plateaus and decide when training should stop."""

	def __init__(
		self,
		enabled: bool = True,
		monitor: str = "val_loss",
		mode: str = "min",
		patience: int = 8,
		min_delta: float = 0.001,
		start_epoch: int = 5,
		stop_on_nan: bool = True,
	) -> None:
		self.enabled = bool(enabled)
		self.monitor = str(monitor)
		self.mode = str(mode).lower()
		if self.mode not in {"min", "max"}:
			raise ValueError(f"EarlyStopping mode must be 'min' or 'max', got {mode!r}.")
		self.patience = max(1, int(patience))
		self.min_delta = max(0.0, float(min_delta))
		self.start_epoch = max(1, int(start_epoch))
		self.stop_on_nan = bool(stop_on_nan)
		self.best_score: float | None = None
		self.best_epoch: int | None = None
		self.num_bad_epochs = 0
		self.should_stop = False
		self.stop_reason = ""
		self.stop_epoch: int | None = None

	def _is_improvement(self, current: float) -> bool:
		if self.best_score is None:
			return True
		if self.mode == "min":
			return current < self.best_score - self.min_delta
		return current > self.best_score + self.min_delta

	def step(self, epoch: int, metrics_dict: Mapping[str, Any]) -> dict[str, Any]:
		"""Update state from one validation check and return loggable fields."""

		epoch_number = int(epoch)
		if not self.enabled:
			return self.log_state()
		if self.monitor not in metrics_dict:
			available = ", ".join(sorted(str(key) for key in metrics_dict.keys()))
			raise KeyError(f"Early stopping monitor {self.monitor!r} not found in metrics. Available metrics: {available}")
		try:
			current = float(metrics_dict[self.monitor])
		except (TypeError, ValueError) as exc:
			raise ValueError(f"Early stopping monitor {self.monitor!r} is not numeric: {metrics_dict[self.monitor]!r}") from exc
		if not math.isfinite(current):
			if self.stop_on_nan:
				self.should_stop = True
				self.stop_epoch = epoch_number
				self.stop_reason = f"Validation metric {self.monitor} is NaN/Inf."
			else:
				self.num_bad_epochs += 1
			return self.log_state()
		if self._is_improvement(current):
			self.best_score = current
			self.best_epoch = epoch_number
			self.num_bad_epochs = 0
		else:
			self.num_bad_epochs += 1
		if epoch_number >= self.start_epoch and self.num_bad_epochs >= self.patience:
			self.should_stop = True
			self.stop_epoch = epoch_number
			self.stop_reason = f"No improvement in {self.monitor} for {self.patience} validation checks."
		return self.log_state()

	def log_state(self) -> dict[str, Any]:
		return {
			"early_stopping_monitor": self.monitor,
			"early_stopping_best_score": self.best_score,
			"early_stopping_best_epoch": self.best_epoch,
			"early_stopping_bad_epochs": self.num_bad_epochs,
			"early_stopping_patience": self.patience,
			"early_stopping_should_stop": bool(self.should_stop),
		}

	def state_dict(self) -> dict[str, Any]:
		return {
			"enabled": self.enabled,
			"monitor": self.monitor,
			"mode": self.mode,
			"patience": self.patience,
			"min_delta": self.min_delta,
			"start_epoch": self.start_epoch,
			"stop_on_nan": self.stop_on_nan,
			"best_score": self.best_score,
			"best_epoch": self.best_epoch,
			"num_bad_epochs": self.num_bad_epochs,
			"stopped_early": bool(self.should_stop),
			"should_stop": bool(self.should_stop),
			"stop_epoch": self.stop_epoch,
			"stop_reason": self.stop_reason,
		}

	def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
		if not isinstance(state, Mapping):
			return
		self.best_score = None if state.get("best_score") in (None, "", "null") else float(state["best_score"])
		best_epoch = state.get("best_epoch")
		self.best_epoch = None if best_epoch in (None, "", "null") else int(best_epoch)
		self.num_bad_epochs = int(state.get("num_bad_epochs", 0))
		self.should_stop = bool(state.get("should_stop", state.get("stopped_early", False)))
		stop_epoch = state.get("stop_epoch")
		self.stop_epoch = None if stop_epoch in (None, "", "null") else int(stop_epoch)
		self.stop_reason = str(state.get("stop_reason", ""))


def build_early_stopping(config: Mapping[str, Any]) -> EarlyStopping:
	training = config.get("training", {}) if isinstance(config.get("training"), Mapping) else {}
	section = training.get("early_stopping", {}) if isinstance(training.get("early_stopping"), Mapping) else {}
	legacy_patience = training.get("early_stopping_patience")
	enabled = bool(section.get("enabled", legacy_patience not in (None, "", "null", 0, 0.0)))
	patience = section.get("patience", legacy_patience if legacy_patience not in (None, "", "null", 0, 0.0) else 8)
	return EarlyStopping(
		enabled=enabled,
		monitor=str(section.get("monitor", "val_loss")),
		mode=str(section.get("mode", "min")),
		patience=int(patience),
		min_delta=float(section.get("min_delta", 0.001)),
		start_epoch=int(section.get("start_epoch", 5)),
		stop_on_nan=bool(section.get("stop_on_nan", True)),
	)
