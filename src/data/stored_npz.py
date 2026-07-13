"""Helpers for memory-mapping arrays stored uncompressed inside ``.npz`` files."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import struct
import zipfile

import numpy as np


@dataclass(frozen=True)
class StoredNpzArrayInfo:
	"""On-disk location and dtype metadata for one ``.npy`` member in an ``.npz``."""

	shape: tuple[int, ...]
	dtype: np.dtype
	fortran_order: bool
	data_offset: int


def _member_name(array_name: str) -> str:
	return array_name if array_name.endswith(".npy") else f"{array_name}.npy"


def _member_payload_offset(path: Path, zip_info: zipfile.ZipInfo) -> int:
	"""Return the byte offset where a stored zip member's payload begins."""

	with path.open("rb") as handle:
		handle.seek(int(zip_info.header_offset))
		local_header = handle.read(30)
	if len(local_header) != 30 or local_header[:4] != b"PK\x03\x04":
		raise ValueError(f"Invalid local zip header for {zip_info.filename!r} in {path}.")
	file_name_length, extra_length = struct.unpack("<HH", local_header[26:30])
	return int(zip_info.header_offset) + 30 + int(file_name_length) + int(extra_length)


def _read_npy_header(path: Path, payload_offset: int) -> StoredNpzArrayInfo:
	"""Parse an ``.npy`` header at ``payload_offset`` inside ``path``."""

	with path.open("rb") as handle:
		handle.seek(int(payload_offset))
		magic = handle.read(6)
		if magic != b"\x93NUMPY":
			raise ValueError(f"Stored NPZ member in {path} does not start with a valid .npy header.")
		version = tuple(handle.read(2))
		if version == (1, 0):
			header_length = struct.unpack("<H", handle.read(2))[0]
		elif version in {(2, 0), (3, 0)}:
			header_length = struct.unpack("<I", handle.read(4))[0]
		else:
			raise ValueError(f"Unsupported .npy version {version!r} in {path}.")
		header_start = handle.tell()
		header = handle.read(int(header_length))
	header_text = header.decode("latin1")
	header_dict = ast.literal_eval(header_text)
	if not isinstance(header_dict, dict):
		raise ValueError(f"Invalid .npy header in {path}.")
	shape_value = header_dict.get("shape")
	if isinstance(shape_value, int):
		shape = (int(shape_value),)
	else:
		shape = tuple(int(value) for value in shape_value)
	dtype = np.dtype(header_dict["descr"])
	if dtype.hasobject:
		raise ValueError(f"Object arrays cannot be memory-mapped safely from {path}.")
	return StoredNpzArrayInfo(
		shape=shape,
		dtype=dtype,
		fortran_order=bool(header_dict.get("fortran_order", False)),
		data_offset=header_start + int(header_length),
	)


def stored_npz_array_info(path: str | Path, array_name: str) -> StoredNpzArrayInfo:
	"""Return array metadata for an uncompressed member without loading its data."""

	resolved_path = Path(path).expanduser().resolve()
	member_name = _member_name(array_name)
	with zipfile.ZipFile(resolved_path) as archive:
		try:
			zip_info = archive.getinfo(member_name)
		except KeyError as exc:
			raise KeyError(f"NPZ archive {resolved_path} is missing member {member_name!r}.") from exc
		if zip_info.compress_type != zipfile.ZIP_STORED:
			raise ValueError(f"NPZ member {member_name!r} in {resolved_path} is compressed and cannot be memory-mapped.")
		if zip_info.flag_bits & 0x1:
			raise ValueError(f"NPZ member {member_name!r} in {resolved_path} is encrypted and cannot be memory-mapped.")
		payload_offset = _member_payload_offset(resolved_path, zip_info)
	return _read_npy_header(resolved_path, payload_offset)


def open_stored_npz_array(path: str | Path, array_name: str) -> np.memmap:
	"""Open an uncompressed NPZ member as a read-only NumPy memmap."""

	resolved_path = Path(path).expanduser().resolve()
	info = stored_npz_array_info(resolved_path, array_name)
	order = "F" if info.fortran_order else "C"
	return np.memmap(
		resolved_path,
		mode="r",
		dtype=info.dtype,
		shape=info.shape,
		offset=info.data_offset,
		order=order,
	)
