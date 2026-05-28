"""ConvLSTM U-Net model assembly for spatiotemporal wildfire prediction."""

from __future__ import annotations

from types import SimpleNamespace

try:
	import torch  # type: ignore[import-not-found]
	import torch.nn as nn  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	nn = SimpleNamespace(Module=object)

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

		self.input_channels = int(input_channels)
		self.output_channels = int(output_channels)
		self.output_activation = output_activation

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

		if self.output_activation == "sigmoid":
			y_pred = torch.sigmoid(y_pred)
		elif self.output_activation == "relu":
			y_pred = torch.relu(y_pred)

		return y_pred


def build_model_from_config(config, input_channels: int):
	"""Build a ``ConvLSTMUNet`` from a configuration dictionary."""

	model_config = config.get("model", config)
	task_type = str(config.get("task_type", model_config.get("task_type", "regression"))).lower()
	output_activation = model_config.get("output_activation")
	if task_type == "multitask":
		output_activation = None
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
	)


if __name__ == "__main__":
	if torch is None:
		print("ConvLSTMUNet smoke test skipped: PyTorch is not installed in this environment")
		raise SystemExit(0)

	x = torch.randn(2, 10, 56, 144, 144)
	model = ConvLSTMUNet(input_channels=56, output_channels=1)
	y = model(x)
	assert y.shape == (2, 1, 144, 144)
	print(y.shape)
