"""Optional physical output constraints for CAWFE-Latte-Lite."""

from __future__ import annotations

from types import SimpleNamespace

try:
	import torch  # type: ignore[import-not-found]
	import torch.nn as nn  # type: ignore[import-not-found]
	import torch.nn.functional as F  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	nn = SimpleNamespace(Module=object)
	F = None


class PhysicalOutputConstraintLayer(nn.Module):
	"""Apply nonnegative constraints to continuous fire outputs."""

	def __init__(
		self,
		constrain_consumed_nonnegative: bool = True,
		constrain_energy_nonnegative: bool = True,
		mask_output_is_logits: bool = True,
	) -> None:
		super().__init__()
		self.constrain_consumed_nonnegative = bool(constrain_consumed_nonnegative)
		self.constrain_energy_nonnegative = bool(constrain_energy_nonnegative)
		self.mask_output_is_logits = bool(mask_output_is_logits)

	def forward(self, pred: torch.Tensor) -> torch.Tensor:
		if pred.ndim != 4:
			raise ValueError(f"PhysicalOutputConstraintLayer expects (B, 4, H, W), got {tuple(pred.shape)}.")
		if int(pred.shape[1]) < 4:
			raise ValueError(f"PhysicalOutputConstraintLayer expects at least 4 channels, got {int(pred.shape[1])}.")
		channels = [pred[:, index : index + 1] for index in range(int(pred.shape[1]))]
		if self.constrain_consumed_nonnegative:
			channels[0] = F.softplus(channels[0])
			channels[1] = F.softplus(channels[1])
		if self.constrain_energy_nonnegative:
			channels[3] = F.softplus(channels[3])
		return torch.cat(channels, dim=1)


__all__ = ["PhysicalOutputConstraintLayer"]
