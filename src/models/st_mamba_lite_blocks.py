"""Building blocks for the in-project ST-Mamba-Lite architecture."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

try:
	import torch  # type: ignore[import-not-found]
	import torch.nn as nn  # type: ignore[import-not-found]
	import torch.nn.functional as F  # type: ignore[import-not-found]
	from torch.utils.checkpoint import checkpoint  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	nn = SimpleNamespace(Module=object, ModuleDict=dict, ModuleList=list)
	F = None
	checkpoint = None

from src.models.earthformer_blocks import StochasticDepth
from src.models.mamba_backend import build_mamba_layer
from src.models.scan_routes import all_supported_routes, canonicalize_route, flatten_by_route, reverse_sequence, unflatten_by_route


def _make_group_norm(num_channels: int) -> nn.GroupNorm:
	"""Build a small-group GroupNorm that always divides ``num_channels``."""

	for group_count in (8, 4, 2, 1):
		if num_channels % group_count == 0:
			return nn.GroupNorm(group_count, num_channels)
	return nn.GroupNorm(1, num_channels)


class ChannelLayerNorm5D(nn.Module):
	"""LayerNorm across the channel dimension of ``(B, T, C, H, W)`` tensors."""

	def __init__(self, channels: int) -> None:
		super().__init__()
		self.channels = int(channels)
		self.norm = nn.LayerNorm(self.channels)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"ChannelLayerNorm5D expects a 5D tensor, got {tuple(x.shape)}.")
		if int(x.shape[2]) != self.channels:
			raise ValueError(f"ChannelLayerNorm5D expected channel dim {self.channels}, got {int(x.shape[2])}.")
		return self.norm(x.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3).contiguous()


class DepthwiseConv3DLocalMixer(nn.Module):
	"""Local spatial-temporal mixing with depthwise 3D convolution."""

	def __init__(self, channels: int, kernel_size: Sequence[int] = (3, 3, 3)) -> None:
		super().__init__()
		if len(kernel_size) != 3:
			raise ValueError(f"depthwise_conv3d_kernel_size must contain 3 integers, got {kernel_size!r}.")
		padding = tuple(int(value) // 2 for value in kernel_size)
		self.channels = int(channels)
		self.depthwise = nn.Conv3d(
			in_channels=self.channels,
			out_channels=self.channels,
			kernel_size=tuple(int(value) for value in kernel_size),
			padding=padding,
			groups=self.channels,
		)
		self.pointwise = nn.Conv3d(self.channels, self.channels, kernel_size=1)
		self.activation = nn.GELU()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"DepthwiseConv3DLocalMixer expects a 5D tensor, got {tuple(x.shape)}.")
		y = x.permute(0, 2, 1, 3, 4).contiguous()
		y = self.depthwise(y)
		y = self.pointwise(y)
		y = self.activation(y)
		return y.permute(0, 2, 1, 3, 4).contiguous()


class ChannelMLP3D(nn.Module):
	"""Channel MLP using 1x1x1 convolutions."""

	def __init__(self, channels: int, mlp_ratio: float, dropout: float = 0.0) -> None:
		super().__init__()
		hidden_channels = max(int(round(float(channels) * float(mlp_ratio))), 1)
		self.net = nn.Sequential(
			nn.Conv3d(channels, hidden_channels, kernel_size=1),
			nn.GELU(),
			nn.Dropout(float(dropout)),
			nn.Conv3d(hidden_channels, channels, kernel_size=1),
			nn.Dropout(float(dropout)),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"ChannelMLP3D expects a 5D tensor, got {tuple(x.shape)}.")
		y = x.permute(0, 2, 1, 3, 4).contiguous()
		y = self.net(y)
		return y.permute(0, 2, 1, 3, 4).contiguous()


class TemporalReadout2D(nn.Module):
	"""Collapse the temporal dimension into one 2D feature map per sample."""

	def __init__(self, channels: int, mode: str = "attention_pool") -> None:
		super().__init__()
		self.channels = int(channels)
		self.mode = str(mode).lower()
		if self.mode not in {"last", "mean", "attention_pool"}:
			raise ValueError(f"Unsupported temporal_readout: {mode!r}.")
		if self.mode == "attention_pool":
			self.scorer = nn.Linear(self.channels, 1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"TemporalReadout2D expects a 5D tensor, got {tuple(x.shape)}.")
		if self.mode == "last":
			return x[:, -1]
		if self.mode == "mean":
			return torch.mean(x, dim=1)
		tokens = x.permute(0, 3, 4, 1, 2).contiguous()
		scores = self.scorer(tokens).squeeze(-1)
		weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
		pooled = torch.sum(tokens * weights, dim=3)
		return pooled.permute(0, 3, 1, 2).contiguous()


class SpatialTemporalRouteMamba(nn.Module):
	"""Route-based spatial-temporal Mamba mixing."""

	def __init__(
		self,
		channels: int,
		mamba_backend: str,
		d_state: int,
		d_conv: int,
		expand: int,
		scan_mode: str,
		scan_routes: Sequence[str] | None,
		bidirectional_scan: bool,
		bidirectional_merge: str = "mean",
		route_merge: str = "mean",
		use_st_mixer: bool = True,
		st_mixer_sequence_order: str = "THW",
		dropout: float = 0.0,
	) -> None:
		super().__init__()
		self.channels = int(channels)
		self.scan_mode = str(scan_mode).lower()
		self.bidirectional_scan = bool(bidirectional_scan)
		self.bidirectional_merge = str(bidirectional_merge).lower()
		self.route_merge = str(route_merge).lower()
		self.use_st_mixer = bool(use_st_mixer)
		self.st_mixer_sequence_order = canonicalize_route(str(st_mixer_sequence_order).replace("W", "V"))
		if self.bidirectional_merge not in {"mean", "sum", "concat_project"}:
			raise ValueError(f"Unsupported bidirectional_merge: {bidirectional_merge!r}.")
		if self.route_merge not in {"mean", "sum", "concat_project"}:
			raise ValueError(f"Unsupported route_merge: {route_merge!r}.")

		if self.scan_mode == "temporal_only":
			self.route_list = []
		elif self.scan_mode == "spatial_temporal_factorized":
			self.route_list = []
		elif self.scan_mode == "route_pair":
			if not scan_routes:
				raise ValueError("scan_routes must be provided for scan_mode='route_pair'.")
			self.route_list = [canonicalize_route(route) for route in scan_routes]
		elif self.scan_mode == "six_route":
			self.route_list = all_supported_routes()
		else:
			raise ValueError(f"Unsupported scan_mode: {scan_mode!r}.")

		self.route_layers = nn.ModuleDict(
			{
				route: build_mamba_layer(
					d_model=self.channels,
					d_state=int(d_state),
					d_conv=int(d_conv),
					expand=int(expand),
					backend=str(mamba_backend),
					dropout=float(dropout),
				)
				for route in self.route_list
			}
		)
		self.temporal_only_layer = None
		self.st_mixer_layer = None
		if self.scan_mode == "temporal_only":
			self.temporal_only_layer = build_mamba_layer(
				d_model=self.channels,
				d_state=int(d_state),
				d_conv=int(d_conv),
				expand=int(expand),
				backend=str(mamba_backend),
				dropout=float(dropout),
			)
		if self.use_st_mixer:
			self.st_mixer_layer = build_mamba_layer(
				d_model=self.channels,
				d_state=int(d_state),
				d_conv=int(d_conv),
				expand=int(expand),
				backend=str(mamba_backend),
				dropout=float(dropout),
			)

		if self.bidirectional_merge == "concat_project":
			self.bidirectional_project = nn.Linear(self.channels * 2, self.channels)
		else:
			self.bidirectional_project = None

		if self.route_merge == "concat_project":
			self.route_project = nn.Conv3d(self.channels * max(1, self._expected_merged_outputs()), self.channels, kernel_size=1)
		else:
			self.route_project = None

		self.backend_name = self._infer_backend_name()

	def _expected_merged_outputs(self) -> int:
		count = 1 if self.scan_mode in {"temporal_only", "spatial_temporal_factorized"} else len(self.route_list)
		if self.use_st_mixer:
			count += 1
		return count

	def _infer_backend_name(self) -> str:
		for layer in list(self.route_layers.values()) + [self.temporal_only_layer, self.st_mixer_layer]:
			if layer is not None:
				return str(getattr(layer, "backend_name", "unknown"))
		return "unknown"

	def _apply_bidirectional(self, layer: nn.Module, sequence: torch.Tensor) -> torch.Tensor:
		forward = layer(sequence)
		if not self.bidirectional_scan:
			return forward
		backward = reverse_sequence(layer(reverse_sequence(sequence)))
		if self.bidirectional_merge == "mean":
			return 0.5 * (forward + backward)
		if self.bidirectional_merge == "sum":
			return forward + backward
		merged = torch.cat([forward, backward], dim=-1)
		return self.bidirectional_project(merged)  # type: ignore[operator]

	def _apply_routes(self, x: torch.Tensor) -> list[torch.Tensor]:
		outputs: list[torch.Tensor] = []
		if self.temporal_only_layer is not None:
			batch_size, time_steps, channels, height, width = tuple(int(value) for value in x.shape)
			sequences = x.permute(0, 3, 4, 1, 2).reshape(batch_size * height * width, time_steps, channels)
			mixed = self._apply_bidirectional(self.temporal_only_layer, sequences)
			outputs.append(mixed.reshape(batch_size, height, width, time_steps, channels).permute(0, 3, 4, 1, 2).contiguous())
		elif self.route_list:
			target_shape = tuple(int(value) for value in x.shape)
			for route in self.route_list:
				sequence = flatten_by_route(x, route)
				mixed = self._apply_bidirectional(self.route_layers[route], sequence)
				outputs.append(unflatten_by_route(mixed, route, target_shape))
		if self.st_mixer_layer is not None:
			sequence = flatten_by_route(x, self.st_mixer_sequence_order)
			mixed = self._apply_bidirectional(self.st_mixer_layer, sequence)
			outputs.append(unflatten_by_route(mixed, self.st_mixer_sequence_order, tuple(int(value) for value in x.shape)))
		return outputs

	def _merge_outputs(self, outputs: list[torch.Tensor]) -> torch.Tensor:
		if not outputs:
			raise ValueError("SpatialTemporalRouteMamba received no outputs to merge.")
		if len(outputs) == 1:
			return outputs[0]
		if self.route_merge == "mean":
			return torch.stack(outputs, dim=0).mean(dim=0)
		if self.route_merge == "sum":
			return torch.stack(outputs, dim=0).sum(dim=0)
		cat = torch.cat(outputs, dim=2).permute(0, 2, 1, 3, 4).contiguous()
		projected = self.route_project(cat)  # type: ignore[operator]
		return projected.permute(0, 2, 1, 3, 4).contiguous()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"SpatialTemporalRouteMamba expects a 5D tensor, got {tuple(x.shape)}.")
		if int(x.shape[2]) != self.channels:
			raise ValueError(f"SpatialTemporalRouteMamba expected channel dim {self.channels}, got {int(x.shape[2])}.")
		outputs = self._apply_routes(x)
		return self._merge_outputs(outputs)


class STMambaBlock(nn.Module):
	"""Residual ST-Mamba-Lite block."""

	def __init__(
		self,
		channels: int,
		mamba_backend: str,
		d_state: int,
		d_conv: int,
		expand: int,
		scan_mode: str,
		scan_routes: Sequence[str] | None,
		bidirectional_scan: bool,
		use_depthwise_conv3d: bool,
		depthwise_conv3d_kernel_size: Sequence[int] = (3, 3, 3),
		use_st_mixer: bool = True,
		st_mixer_sequence_order: str = "THW",
		dropout: float = 0.0,
		drop_path: float = 0.0,
		mlp_ratio: float = 4.0,
		gradient_checkpointing: bool = False,
	) -> None:
		super().__init__()
		self.channels = int(channels)
		self.gradient_checkpointing = bool(gradient_checkpointing)

		self.norm1 = ChannelLayerNorm5D(self.channels)
		self.local_mixer = (
			DepthwiseConv3DLocalMixer(self.channels, kernel_size=depthwise_conv3d_kernel_size)
			if bool(use_depthwise_conv3d)
			else nn.Identity()
		)
		self.norm2 = ChannelLayerNorm5D(self.channels)
		self.route_mamba = SpatialTemporalRouteMamba(
			channels=self.channels,
			mamba_backend=str(mamba_backend),
			d_state=int(d_state),
			d_conv=int(d_conv),
			expand=int(expand),
			scan_mode=str(scan_mode),
			scan_routes=scan_routes,
			bidirectional_scan=bool(bidirectional_scan),
			use_st_mixer=bool(use_st_mixer),
			st_mixer_sequence_order=str(st_mixer_sequence_order),
			dropout=float(dropout),
		)
		self.norm3 = ChannelLayerNorm5D(self.channels)
		self.channel_mlp = ChannelMLP3D(self.channels, mlp_ratio=float(mlp_ratio), dropout=float(dropout))
		self.drop_path = StochasticDepth(float(drop_path))
		self.backend_name = self.route_mamba.backend_name

	def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
		x = x + self.drop_path(self.local_mixer(self.norm1(x)))
		x = x + self.drop_path(self.route_mamba(self.norm2(x)))
		x = x + self.drop_path(self.channel_mlp(self.norm3(x)))
		return x

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"STMambaBlock expects a 5D tensor, got {tuple(x.shape)}.")
		if int(x.shape[2]) != self.channels:
			raise ValueError(f"STMambaBlock expected channel dim {self.channels}, got {int(x.shape[2])}.")
		if self.gradient_checkpointing and self.training and checkpoint is not None:
			return checkpoint(self._forward_impl, x, use_reentrant=False)
		return self._forward_impl(x)
