"""Reusable ConvLSTM building blocks for sequence-to-map forecasting.

Expected input shape for the stacked module is ``(B, T, C, H, W)`` when
``batch_first=True``.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple
from types import SimpleNamespace

try:
	import torch  # type: ignore[import-not-found]
	import torch.nn as nn  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	nn = SimpleNamespace(Module=object, Conv2d=object, ModuleList=list)


def _to_list(value, num_layers: int) -> List:
	"""Broadcast a scalar or validate a sequence against ``num_layers``."""

	if isinstance(value, (list, tuple)):
		if len(value) != num_layers:
			raise ValueError(f"Expected {num_layers} values, got {len(value)}.")
		return list(value)
	return [value for _ in range(num_layers)]


class ConvLSTMCell(nn.Module):
	"""Single ConvLSTM cell.

	Args:
		input_dim: Number of channels in the input tensor ``(B, C, H, W)``.
		hidden_dim: Number of channels in the hidden state.
		kernel_size: Convolution kernel size used for the gated update.
		bias: Whether to include a bias term in the convolution.

	Shapes:
		input_tensor: ``(B, C, H, W)``
		h_cur: ``(B, hidden_dim, H, W)``
		c_cur: ``(B, hidden_dim, H, W)``
		h_next / c_next: ``(B, hidden_dim, H, W)``
	"""

	def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int, bias: bool = True) -> None:
		super().__init__()
		if input_dim <= 0:
			raise ValueError(f"input_dim must be positive, got {input_dim}.")
		if hidden_dim <= 0:
			raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
		if kernel_size <= 0:
			raise ValueError(f"kernel_size must be positive, got {kernel_size}.")

		self.input_dim = int(input_dim)
		self.hidden_dim = int(hidden_dim)
		self.kernel_size = int(kernel_size)
		self.bias = bool(bias)

		padding = self.kernel_size // 2
		self.conv = nn.Conv2d(
			in_channels=self.input_dim + self.hidden_dim,
			out_channels=4 * self.hidden_dim,
			kernel_size=self.kernel_size,
			padding=padding,
			bias=self.bias,
		)

	def forward(self, input_tensor: torch.Tensor, cur_state: Tuple[torch.Tensor, torch.Tensor]):
		"""Run one ConvLSTM step."""

		h_cur, c_cur = cur_state
		if input_tensor.ndim != 4:
			raise ValueError(f"Expected input_tensor to have 4 dimensions, got {tuple(input_tensor.shape)}.")
		if h_cur.ndim != 4 or c_cur.ndim != 4:
			raise ValueError("Hidden and cell states must be 4D tensors shaped (B, hidden_dim, H, W).")
		if input_tensor.shape[0] != h_cur.shape[0] or input_tensor.shape[0] != c_cur.shape[0]:
			raise ValueError("Batch size of input_tensor, h_cur, and c_cur must match.")
		if input_tensor.shape[2:] != h_cur.shape[2:] or input_tensor.shape[2:] != c_cur.shape[2:]:
			raise ValueError("Spatial dimensions of input_tensor, h_cur, and c_cur must match.")
		if input_tensor.shape[1] != self.input_dim:
			raise ValueError(
				f"Expected input_tensor to have {self.input_dim} channels, got {input_tensor.shape[1]}."
			)
		if h_cur.shape[1] != self.hidden_dim or c_cur.shape[1] != self.hidden_dim:
			raise ValueError(
				f"Expected hidden and cell states to have {self.hidden_dim} channels, got "
				f"h={h_cur.shape[1]}, c={c_cur.shape[1]}."
			)

		combined = torch.cat([input_tensor, h_cur], dim=1)
		gates = self.conv(combined)
		i_gate, f_gate, o_gate, g_gate = torch.chunk(gates, chunks=4, dim=1)

		i_gate = torch.sigmoid(i_gate)
		f_gate = torch.sigmoid(f_gate)
		o_gate = torch.sigmoid(o_gate)
		g_gate = torch.tanh(g_gate)

		c_next = f_gate * c_cur + i_gate * g_gate
		h_next = o_gate * torch.tanh(c_next)
		return h_next, c_next

	def init_hidden(self, batch_size: int, spatial_size: Tuple[int, int], device=None, dtype=None):
		"""Initialize hidden and cell states to zeros on the requested device/dtype."""

		height, width = spatial_size
		if batch_size <= 0:
			raise ValueError(f"batch_size must be positive, got {batch_size}.")
		if height <= 0 or width <= 0:
			raise ValueError(f"spatial_size must contain positive values, got {spatial_size}.")

		h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device, dtype=dtype)
		c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device, dtype=dtype)
		return h, c


class ConvLSTM(nn.Module):
	"""Stacked ConvLSTM network.

	Args:
		input_dim: Number of channels in the input tensor.
		hidden_dim: Either an int or a list of ints, one per layer.
		kernel_size: Either an int or a list of ints, one per layer.
		num_layers: Number of stacked ConvLSTM layers.
		batch_first: Expect input shape ``(B, T, C, H, W)`` when ``True``.
		bias: Whether to include bias terms in the gated convolutions.
		return_all_layers: If ``True``, return outputs/states for every layer.

	Returns when ``return_all_layers=False``:
		last_layer_output: ``(B, T, hidden_dim[-1], H, W)``
		last_states: list containing ``(h, c)`` for the final layer
	"""

	def __init__(
		self,
		input_dim: int,
		hidden_dim,
		kernel_size,
		num_layers: int,
		batch_first: bool = True,
		bias: bool = True,
		return_all_layers: bool = False,
	) -> None:
		super().__init__()
		if input_dim <= 0:
			raise ValueError(f"input_dim must be positive, got {input_dim}.")
		if num_layers <= 0:
			raise ValueError(f"num_layers must be positive, got {num_layers}.")

		self.input_dim = int(input_dim)
		self.hidden_dim = _to_list(hidden_dim, num_layers)
		self.kernel_size = _to_list(kernel_size, num_layers)
		self.num_layers = int(num_layers)
		self.batch_first = bool(batch_first)
		self.bias = bool(bias)
		self.return_all_layers = bool(return_all_layers)

		cells = []
		for layer_index in range(self.num_layers):
			current_input_dim = self.input_dim if layer_index == 0 else self.hidden_dim[layer_index - 1]
			cells.append(
				ConvLSTMCell(
					input_dim=current_input_dim,
					hidden_dim=self.hidden_dim[layer_index],
					kernel_size=self.kernel_size[layer_index],
					bias=self.bias,
				)
			)
		self.cell_list = nn.ModuleList(cells)

	def forward(self, input_tensor: torch.Tensor, hidden_state=None):
		"""Process a 5D input tensor shaped ``(B, T, C, H, W)`` or ``(T, B, C, H, W)``."""

		if input_tensor.ndim != 5:
			raise ValueError(f"ConvLSTM expects a 5D tensor, got shape {tuple(input_tensor.shape)}.")

		if not self.batch_first:
			input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

		batch_size, seq_len, _, height, width = input_tensor.shape
		if seq_len <= 0:
			raise ValueError("Sequence length must be positive.")

		if hidden_state is None:
			hidden_state = self._init_hidden(batch_size=batch_size, spatial_size=(height, width), device=input_tensor.device, dtype=input_tensor.dtype)
		else:
			if len(hidden_state) != self.num_layers:
				raise ValueError(f"Expected hidden_state for {self.num_layers} layers, got {len(hidden_state)}.")

		layer_output_list = []
		last_state_list = []
		current_input = input_tensor

		for layer_index, cell in enumerate(self.cell_list):
			h, c = hidden_state[layer_index]
			outputs = []
			for time_index in range(seq_len):
				h, c = cell(current_input[:, time_index, :, :, :], (h, c))
				outputs.append(h)

			layer_output = torch.stack(outputs, dim=1)
			current_input = layer_output
			layer_output_list.append(layer_output)
			last_state_list.append((h, c))

		if self.return_all_layers:
			return layer_output_list, last_state_list

		return layer_output_list[-1], last_state_list

	def _init_hidden(self, batch_size: int, spatial_size: Tuple[int, int], device=None, dtype=None):
		"""Initialize each layer's hidden state."""

		return [cell.init_hidden(batch_size, spatial_size, device=device, dtype=dtype) for cell in self.cell_list]


if __name__ == "__main__":
	if torch is None:
		print("ConvLSTM smoke test skipped: PyTorch is not installed in this environment")
		raise SystemExit(0)

	model = ConvLSTM(
		input_dim=56,
		hidden_dim=32,
		kernel_size=3,
		num_layers=1,
		batch_first=True,
		bias=True,
		return_all_layers=False,
	)
	x = torch.randn(2, 10, 56, 144, 144)
	output, states = model(x)
	print("output shape:", tuple(output.shape))
