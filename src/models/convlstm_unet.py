"""ConvLSTM U-Net model assembly for spatiotemporal wildfire prediction."""

from __future__ import annotations

from types import SimpleNamespace
import warnings

try:
	import torch  # type: ignore[import-not-found]
	import torch.nn as nn  # type: ignore[import-not-found]
	import torch.nn.functional as F  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	nn = SimpleNamespace(Module=object)
	F = None

from src.models.convlstm import ConvLSTM
from src.models.unet import UNet2D


class ConvLSTMUNet(nn.Module):
	"""Baseline ConvLSTM + U-Net model.

	Input shape:
		``(B, T, C, H, W)``

	Output shape:
		``(B, output_channels, H, W)``

	The model first encodes the temporal dimension with a ConvLSTM. The last
	hidden map from the final ConvLSTM layer is then passed through a 2D U-Net
	to produce the final spatial prediction.
	"""

	def __init__(
		self,
		input_channels: int,
		output_channels: int = 1,
		convlstm_hidden_dim: int = 64,
		convlstm_kernel_size: int = 3,
		convlstm_num_layers: int = 1,
		unet_base_channels: int = 64,
		unet_depth: int = 4,
		dropout: float = 0.0,
		output_activation: str | None = None,
		use_mask_gated_regression: bool = False,
		regression_activation: str | None = "relu",
		mask_gate_source: str = "predicted_mask",
		mask_gate_mode: str = "soft",
		mask_gate_threshold: float = 0.5,
		detach_mask_gate: bool = False,
		mask_gate_min: float = 0.0,
		output_bias_init: dict | None = None,
	) -> None:
		super().__init__()
		if input_channels <= 0:
			raise ValueError(f"input_channels must be positive, got {input_channels}.")
		if output_channels <= 0:
			raise ValueError(f"output_channels must be positive, got {output_channels}.")
		if convlstm_hidden_dim <= 0:
			raise ValueError(f"convlstm_hidden_dim must be positive, got {convlstm_hidden_dim}.")
		if convlstm_kernel_size <= 0:
			raise ValueError(f"convlstm_kernel_size must be positive, got {convlstm_kernel_size}.")
		if convlstm_num_layers <= 0:
			raise ValueError(f"convlstm_num_layers must be positive, got {convlstm_num_layers}.")
		if unet_base_channels <= 0:
			raise ValueError(f"unet_base_channels must be positive, got {unet_base_channels}.")
		if unet_depth < 1:
			raise ValueError(f"unet_depth must be at least 1, got {unet_depth}.")
		if not 0.0 <= dropout < 1.0:
			raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
		if output_activation not in (None, "sigmoid", "relu"):
			raise ValueError(
				"output_activation must be one of None, 'sigmoid', or 'relu', "
				f"got {output_activation!r}."
			)
		regression_activation = None if regression_activation is None else str(regression_activation).lower()
		if regression_activation not in ("relu", "softplus", "none", None):
			raise ValueError(
				"regression_activation must be one of 'relu', 'softplus', 'none', or None, "
				f"got {regression_activation!r}."
			)
		mask_gate_source = str(mask_gate_source).lower()
		if mask_gate_source != "predicted_mask":
			raise ValueError(f"mask_gate_source must be 'predicted_mask', got {mask_gate_source!r}.")
		mask_gate_mode = str(mask_gate_mode).lower()
		if mask_gate_mode not in ("soft", "hard"):
			raise ValueError(f"mask_gate_mode must be 'soft' or 'hard', got {mask_gate_mode!r}.")
		if not 0.0 <= float(mask_gate_threshold) <= 1.0:
			raise ValueError(f"mask_gate_threshold must be in [0, 1], got {mask_gate_threshold}.")
		if not 0.0 <= float(mask_gate_min) <= 1.0:
			raise ValueError(f"mask_gate_min must be in [0, 1], got {mask_gate_min}.")
		if bool(use_mask_gated_regression) and int(output_channels) != 4:
			raise ValueError("Mask-gated regression requires output_channels=4 for multitask ConvLSTM outputs.")

		self.input_channels = int(input_channels)
		self.output_channels = int(output_channels)
		self.output_activation = output_activation
		self.use_mask_gated_regression = bool(use_mask_gated_regression)
		self.regression_activation = regression_activation or "none"
		self.mask_gate_source = mask_gate_source
		self.mask_gate_mode = mask_gate_mode
		self.mask_gate_threshold = float(mask_gate_threshold)
		self.detach_mask_gate = bool(detach_mask_gate)
		self.mask_gate_min = float(mask_gate_min)
		self.output_bias_init_config = dict(output_bias_init or {})
		self.output_bias_init_applied = False

		self.temporal_encoder = ConvLSTM(
			input_dim=self.input_channels,
			hidden_dim=convlstm_hidden_dim,
			kernel_size=convlstm_kernel_size,
			num_layers=convlstm_num_layers,
			batch_first=True,
			bias=True,
			return_all_layers=False,
		)
		self.spatial_decoder = UNet2D(
			in_channels=convlstm_hidden_dim,
			out_channels=self.output_channels,
			base_channels=unet_base_channels,
			depth=unet_depth,
			bilinear=True,
			dropout=dropout,
		)
		self._initialize_output_bias()

	def _apply_regression_activation(self, x: torch.Tensor) -> torch.Tensor:
		if self.regression_activation == "relu":
			return F.relu(x)
		if self.regression_activation == "softplus":
			return F.softplus(x)
		if self.regression_activation == "none":
			return x
		raise ValueError(f"Unsupported regression_activation: {self.regression_activation!r}.")

	def _apply_mask_gated_outputs(self, raw: torch.Tensor) -> torch.Tensor:
		if raw.ndim != 4 or int(raw.shape[1]) != 4:
			raise ValueError(f"Mask-gated ConvLSTM output expects raw shape (B,4,H,W), got {tuple(raw.shape)}.")
		raw_surface = raw[:, 0:1]
		raw_canopy = raw[:, 1:2]
		mask_logits = raw[:, 2:3]
		raw_energy = raw[:, 3:4]
		fire_prob = torch.sigmoid(mask_logits)
		if self.mask_gate_mode == "soft":
			gate = fire_prob
		elif self.mask_gate_mode == "hard":
			gate = (fire_prob > self.mask_gate_threshold).to(dtype=raw.dtype)
		else:
			raise ValueError(f"Unsupported mask_gate_mode: {self.mask_gate_mode!r}.")
		if self.detach_mask_gate:
			gate = gate.detach()
		if self.mask_gate_min > 0.0:
			gate = self.mask_gate_min + (1.0 - self.mask_gate_min) * gate
		surface = self._apply_regression_activation(raw_surface) * gate
		canopy = self._apply_regression_activation(raw_canopy) * gate
		energy = self._apply_regression_activation(raw_energy) * gate
		return torch.cat([surface, canopy, mask_logits, energy], dim=1)

	def _initialize_output_bias(self) -> None:
		config = self.output_bias_init_config
		if not bool(config.get("enabled", False)):
			return
		final_conv = getattr(getattr(self.spatial_decoder, "outc", None), "proj", None)
		if final_conv is None or not isinstance(final_conv, nn.Conv2d):
			warnings.warn("ConvLSTM output_bias_init requested, but final Conv2d layer was not found.", RuntimeWarning)
			return
		if final_conv.bias is None:
			warnings.warn("ConvLSTM output_bias_init requested, but final Conv2d layer has no bias.", RuntimeWarning)
			return
		if int(final_conv.out_channels) < 4:
			warnings.warn("ConvLSTM output_bias_init requested, but final Conv2d has fewer than 4 channels.", RuntimeWarning)
			return
		with torch.no_grad():
			final_conv.bias[0] = float(config.get("regression_bias", -2.0))
			final_conv.bias[1] = float(config.get("regression_bias", -2.0))
			final_conv.bias[2] = float(config.get("mask_bias", -2.0))
			final_conv.bias[3] = float(config.get("energy_bias", -2.0))
		self.output_bias_init_applied = True

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Predict a 2D target map from a sequence of input maps.

		Args:
			x: Input tensor shaped ``(B, T, C, H, W)``.

		Returns:
			A tensor shaped ``(B, output_channels, H, W)``.
		"""

		if x.ndim != 5:
			raise ValueError(f"ConvLSTMUNet expects a 5D tensor, got shape {tuple(x.shape)}.")
		if x.shape[2] != self.input_channels:
			raise ValueError(f"Expected {self.input_channels} input channels, got {x.shape[2]}.")

		temporal_output, _ = self.temporal_encoder(x)
		h_last = temporal_output[:, -1]
		y_pred = self.spatial_decoder(h_last)

		if self.use_mask_gated_regression:
			y_pred = self._apply_mask_gated_outputs(y_pred)
		elif self.output_activation == "sigmoid":
			y_pred = torch.sigmoid(y_pred)
		elif self.output_activation == "relu":
			y_pred = torch.relu(y_pred)

		return y_pred


def build_convlstm_unet_from_config(config, input_channels: int):
	"""Build a ``ConvLSTMUNet`` from a configuration dictionary."""

	model_config = config.get("model", config)
	convlstm_config = config.get("convlstm_unet", {})
	if not isinstance(convlstm_config, dict):
		convlstm_config = {}
	task_type = str(config.get("task_type", model_config.get("task_type", "regression"))).lower()
	output_activation = model_config.get("output_activation")
	if task_type == "multitask":
		output_activation = None
	def _convlstm_value(name, default=None):
		return convlstm_config.get(name, model_config.get(name, default))

	return ConvLSTMUNet(
		input_channels=input_channels,
		output_channels=int(model_config.get("output_channels", 1)),
		convlstm_hidden_dim=int(model_config.get("convlstm_hidden_dim", model_config.get("hidden_dim", 64))),
		convlstm_kernel_size=int(model_config.get("convlstm_kernel_size", model_config.get("kernel_size", 3))),
		convlstm_num_layers=int(model_config.get("convlstm_num_layers", model_config.get("num_layers", 1))),
		unet_base_channels=int(model_config.get("unet_base_channels", 64)),
		unet_depth=int(model_config.get("unet_depth", 4)),
		dropout=float(model_config.get("dropout", 0.0)),
		output_activation=output_activation,
		use_mask_gated_regression=bool(_convlstm_value("use_mask_gated_regression", False)),
		regression_activation=_convlstm_value("regression_activation", "relu"),
		mask_gate_source=str(_convlstm_value("mask_gate_source", "predicted_mask")),
		mask_gate_mode=str(_convlstm_value("mask_gate_mode", "soft")),
		mask_gate_threshold=float(_convlstm_value("mask_gate_threshold", 0.5)),
		detach_mask_gate=bool(_convlstm_value("detach_mask_gate", False)),
		mask_gate_min=float(_convlstm_value("mask_gate_min", 0.0)),
		output_bias_init=_convlstm_value("output_bias_init", {}),
	)


def build_model_from_config(config, input_channels: int):
	"""Backward-compatible generic model builder entry point."""

	from src.models.model_factory import build_model_from_config as _build_model_from_config

	return _build_model_from_config(config, input_channels=input_channels)


if __name__ == "__main__":
	if torch is None:
		print("ConvLSTMUNet smoke test skipped: PyTorch is not installed in this environment")
		raise SystemExit(0)

	x = torch.randn(2, 10, 56, 144, 144)
	model = ConvLSTMUNet(input_channels=56, output_channels=1)
	y = model(x)
	assert y.shape == (2, 1, 144, 144)
	print(y.shape)
