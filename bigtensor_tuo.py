import argparse
import os

import numpy as np


DEFAULT_ASC_SHAPE = (72, 72, 46, 10)
DEFAULT_FLUX_SHAPE = (72, 72, 4)
DEFAULT_FUEL_SHAPE = (360, 360, 2)
DEFAULT_POOL_WINDOW = (5, 5)


def _load_flat_text_data(input_filepath, data_type=np.float32, skip_header_rows=0):
    """Load a text file into a flat numeric array."""
    return np.loadtxt(input_filepath, dtype=data_type, skiprows=skip_header_rows)


def _non_finite_counts(array):
    """Return NaN and Inf counts for a numeric array."""
    nan_count = int(np.isnan(array).sum())
    inf_count = int(np.isinf(array).sum())
    return nan_count, inf_count


def _raise_if_non_finite(array, stage_name, input_filepath):
    """Fail fast when non-finite values are present."""
    nan_count, inf_count = _non_finite_counts(array)
    if nan_count == 0 and inf_count == 0:
        return

    raise ValueError(
        f"{stage_name} contains non-finite values for '{input_filepath}': "
        f"nan_count={nan_count}, inf_count={inf_count}."
    )


def _print_tensor_summary(array, tensor_name):
    """Print a compact numeric summary for tracing dataset generation."""
    finite_mask = np.isfinite(array)
    finite_count = int(finite_mask.sum())
    nan_count, inf_count = _non_finite_counts(array)
    print(
        f"INFO {tensor_name}: shape={array.shape}, dtype={array.dtype}, "
        f"finite_count={finite_count}, nan_count={nan_count}, inf_count={inf_count}"
    )
    if finite_count == 0:
        return

    finite_values = array[finite_mask]
    print(
        f"INFO {tensor_name}: min={float(finite_values.min()):.6g}, "
        f"max={float(finite_values.max()):.6g}, mean={float(finite_values.mean()):.6g}"
    )


def _resolve_data_path(data_root, filename):
    """Join a data-root directory and a filename."""
    return os.path.join(data_root, filename)


def _pool_2d_channel(channel_2d, pool_window, pooling_mode):
    """Pool a single 2D channel using non-overlapping windows."""
    h, w = channel_2d.shape
    ph, pw = pool_window

    if h % ph != 0 or w % pw != 0:
        raise ValueError("Channel shape is not divisible by the pooling window size.")

    reshaped_for_pooling = channel_2d.reshape(h // ph, ph, w // pw, pw)

    if pooling_mode == "average":
        return reshaped_for_pooling.mean(axis=(1, 3))
    if pooling_mode == "max":
        return reshaped_for_pooling.max(axis=(1, 3))
    if pooling_mode == "sum":
        return reshaped_for_pooling.sum(axis=(1, 3))
    raise ValueError("pooling_mode must be one of: average, max, sum.")


def average_pool_fuel(
    input_filepath,
    initial_shape=(360, 360, 2),
    pool_window=(5, 5),
    data_type=np.float32,
    skip_header_rows=1,
    pooling_mode="average",
):
    """
    Reads data from a text file, reshapes it, and applies 2D pooling.

    Args:
        input_filepath (str): The path to the input text file.
        initial_shape (tuple): The 3D shape of the data before pooling.
        pool_window (tuple): The (height, width) of the pooling window.
        data_type (np.dtype, optional): The numpy data type. Defaults to np.float32.
        skip_header_rows (int, optional): Number of header lines to skip. Defaults to 0.

    Returns:
        np.ndarray: The new downsampled tensor, or None if an error occurred.
    """
    try:
        print(f"⏳ Loading data from '{input_filepath}'...")
        flat_data = _load_flat_text_data(
            input_filepath=input_filepath,
            data_type=data_type,
            skip_header_rows=skip_header_rows,
        )
        _raise_if_non_finite(flat_data, "Raw fuel text data", input_filepath)

        h, w, c = initial_shape
        if c != 2:
            raise ValueError(f"Fuel tensors are expected to have exactly 2 channels, got {c}.")

        channel_size = h * w
        required_elements = channel_size * c
        if flat_data.size < required_elements:
            raise ValueError(
                f"File requires {required_elements} elements, but only found {flat_data.size}."
            )

        surface_flat = flat_data[:channel_size]
        canopy_flat = flat_data[channel_size:required_elements]

        surface_tensor = surface_flat.reshape((h, w), order="C")
        canopy_tensor = canopy_flat.reshape((h, w), order="C")
        print(f"✅ Fuel channels reshaped to: surface={surface_tensor.shape}, canopy={canopy_tensor.shape}")
        _raise_if_non_finite(surface_tensor, "Surface fuel tensor after reshape", input_filepath)
        _raise_if_non_finite(canopy_tensor, "Canopy fuel tensor after reshape", input_filepath)

        print(f"⏳ Performing {pool_window[0]}x{pool_window[1]} {pooling_mode} pooling...")
        pooled_surface = _pool_2d_channel(surface_tensor, pool_window, pooling_mode)
        pooled_canopy = _pool_2d_channel(canopy_tensor, pool_window, pooling_mode)
        pooled_tensor = np.stack((pooled_surface, pooled_canopy), axis=2)

        _raise_if_non_finite(pooled_tensor, "Fuel tensor after pooling", input_filepath)
        _print_tensor_summary(pooled_tensor, "fuel_tensor")

        print("\n🎉 Success! Pooling complete.")
        print(f"Final shape of pooled tensor: {pooled_tensor.shape}")

        return pooled_tensor

    except FileNotFoundError:
        print(f"❌ Error: The file '{input_filepath}' was not found.")
        return None


def load_asc_tensor(
    input_filepath,
    keep_z_levels=46,
    target_shape=DEFAULT_ASC_SHAPE,
    data_type=np.float32,
    skip_header_rows=0,
    allow_padding=True,
):
    """Load the atmospheric ASC file into a 4D tensor."""
    try:
        x_size, y_size, max_z_levels, field_count = target_shape
        if keep_z_levels < 1 or keep_z_levels > max_z_levels:
            raise ValueError(f"keep_z_levels must be between 1 and {max_z_levels}, got {keep_z_levels}.")

        channel_size = x_size * y_size
        z_level_size = channel_size * field_count
        total_elements = z_level_size * keep_z_levels
        print(f"⏳ Loading ASC data from '{input_filepath}'...")
        flat_data = _load_flat_text_data(
            input_filepath=input_filepath,
            data_type=data_type,
            skip_header_rows=skip_header_rows,
        )
        _raise_if_non_finite(flat_data, "Raw ASC text data", input_filepath)

        if flat_data.size < total_elements:
            missing = total_elements - flat_data.size
            if not allow_padding:
                raise ValueError(
                    f"File has {flat_data.size} numbers, but {total_elements} are needed for keep_z_levels={keep_z_levels}."
                )
            print(f"⚠️  ASC file is short by {missing} value(s); padding the tail with zeros.")
            flat_data = np.pad(flat_data, (0, missing), mode='constant', constant_values=0)

        asc_levels = []
        for z_index in range(keep_z_levels):
            z_start = z_index * z_level_size
            z_end = z_start + z_level_size
            z_flat = flat_data[z_start:z_end]

            level_fields = []
            for field_index in range(field_count):
                field_start = field_index * channel_size
                field_end = field_start + channel_size
                field_tensor = z_flat[field_start:field_end].reshape((x_size, y_size), order="C")
                _raise_if_non_finite(
                    field_tensor,
                    f"ASC field {field_index + 1} tensor at z-level {z_index + 1} after reshape",
                    input_filepath,
                )
                level_fields.append(field_tensor)

            asc_levels.append(np.stack(level_fields, axis=2))

        asc_tensor = np.stack(asc_levels, axis=2)
        _raise_if_non_finite(asc_tensor, "ASC tensor after reshape", input_filepath)
        _print_tensor_summary(asc_tensor, "asc_tensor")
        print(f"✅ ASC tensor shape: {asc_tensor.shape}")
        return asc_tensor

    except FileNotFoundError:
        print(f"❌ Error: The file '{input_filepath}' was not found.")
        return None
    except Exception as e:
        print(f"❌ An error occurred while loading ASC data: {e}")
        return None


def load_flux_tensor(
    input_filepath,
    target_shape=DEFAULT_FLUX_SHAPE,
    data_type=np.float32,
    skip_header_rows=1,
):
    """Load the flux file into a 3D tensor with one channel per flux field."""
    try:
        h, w, c = target_shape
        if c != 4:
            raise ValueError(f"Flux tensors are expected to have exactly 4 channels, got {c}.")

        channel_size = h * w
        total_elements = channel_size * c
        print(f"⏳ Loading flux data from '{input_filepath}'...")
        flat_data = _load_flat_text_data(
            input_filepath=input_filepath,
            data_type=data_type,
            skip_header_rows=skip_header_rows,
        )
        _raise_if_non_finite(flat_data, "Raw flux text data", input_filepath)

        if flat_data.size < total_elements:
            raise ValueError(
                f"Flux file has {flat_data.size} numbers, but {total_elements} are needed for shape {target_shape}."
            )

        flux_channels = []
        for channel_index in range(c):
            start = channel_index * channel_size
            end = start + channel_size
            channel_tensor = flat_data[start:end].reshape((h, w), order="C")
            _raise_if_non_finite(
                channel_tensor,
                f"Flux channel {channel_index + 1} tensor after reshape",
                input_filepath,
            )
            flux_channels.append(channel_tensor)

        flux_tensor = np.stack(flux_channels, axis=2)
        _raise_if_non_finite(flux_tensor, "Flux tensor after reshape", input_filepath)
        _print_tensor_summary(flux_tensor, "flux_tensor")
        print(f"✅ Flux tensor shape: {flux_tensor.shape}")
        return flux_tensor

    except FileNotFoundError:
        print(f"❌ Error: The file '{input_filepath}' was not found.")
        return None
    except Exception as e:
        print(f"❌ An error occurred while loading flux data: {e}")
        return None


def merge_timestamp(
    base_file,
    flux_file,
    fuel_file,
    keep_z_levels=46,
    asc_shape=DEFAULT_ASC_SHAPE,
    flux_shape=DEFAULT_FLUX_SHAPE,
    fuel_shape=DEFAULT_FUEL_SHAPE,
    pool_window=DEFAULT_POOL_WINDOW,
    data_type=np.float32,
    asc_skip_header_rows=0,
    flux_skip_header_rows=1,
    fuel_skip_header_rows=1,
    fuel_pooling_mode="average",
):
    """Merge one timestamp into a single ML-ready tensor.

    The ASC contribution preserves all 10 variables for each retained z-level,
    so the output shape is:

        (72, 72, keep_z_levels * 10 + 4 + 2)
    """
    try:
        asc_tensor = load_asc_tensor(
            input_filepath=base_file,
            keep_z_levels=keep_z_levels,
            target_shape=asc_shape,
            data_type=data_type,
            skip_header_rows=asc_skip_header_rows,
        )
        if asc_tensor is None:
            return None

        asc_features = np.transpose(asc_tensor, (0, 1, 3, 2)).reshape(
            asc_tensor.shape[0], asc_tensor.shape[1], keep_z_levels * asc_tensor.shape[3]
        )
        _raise_if_non_finite(asc_features, "Flattened ASC feature tensor", base_file)
        _print_tensor_summary(asc_features, "asc_features")
        print(f"✅ ASC features flattened to shape: {asc_features.shape}")

        flux_tensor = load_flux_tensor(
            input_filepath=flux_file,
            target_shape=flux_shape,
            data_type=data_type,
            skip_header_rows=flux_skip_header_rows,
        )
        if flux_tensor is None:
            return None

        fuel_tensor = average_pool_fuel(
            input_filepath=fuel_file,
            initial_shape=fuel_shape,
            pool_window=pool_window,
            data_type=data_type,
            skip_header_rows=fuel_skip_header_rows,
            pooling_mode=fuel_pooling_mode,
        )
        if fuel_tensor is None:
            return None

        merged_tensor = np.concatenate((asc_features, flux_tensor, fuel_tensor), axis=2)
        merged_tensor = merged_tensor.astype(data_type, copy=False)
        _raise_if_non_finite(merged_tensor, "Merged tensor before save", base_file)
        _print_tensor_summary(merged_tensor, "merged_tensor")
        print(f"🎉 Merged tensor shape: {merged_tensor.shape}")
        return merged_tensor

    except Exception as e:
        print(f"❌ An error occurred while merging timestamp data: {e}")
        return None


def process_all_files(
    keep_z_levels_list=(5,),
    start_ts=286,
    end_ts=1999,
    output_root="./final",
    fuel_pooling_mode="sum",
    data_root="./data",
):
    """Generate several dataset versions, one per requested z-depth."""
    if isinstance(keep_z_levels_list, int):
        keep_z_levels_list = (keep_z_levels_list,)

    if not os.path.exists(output_root):
        os.makedirs(output_root)
        print(f"✅ Created output directory: '{output_root}'")

    for keep_z_levels in keep_z_levels_list:
        version_dir = os.path.join(output_root, f"keepz_{keep_z_levels:02d}_tuo")
        os.makedirs(version_dir, exist_ok=True)
        print(f"\n🧭 Writing dataset version for keep_z_levels={keep_z_levels} to '{version_dir}'")

        for ts in range(start_ts, end_ts + 1):
            print(f"\n{'='*20} PROCESSING TIMESTAMP {ts:04d} {'='*20}")

            base_file = _resolve_data_path(data_root, f"FRFCA1M04ASC.{ts:04d}")
            flux_file = _resolve_data_path(data_root, f"FRFCA1M04ASC.flux.{ts:04d}")
            fuel_file = _resolve_data_path(data_root, f"FRFCA1M04ASC.fuel.{ts:04d}")
            output_file = os.path.join(version_dir, f"tensor{ts:04d}.npy")

            merged_tensor = merge_timestamp(
                base_file=base_file,
                flux_file=flux_file,
                fuel_file=fuel_file,
                keep_z_levels=keep_z_levels,
                fuel_pooling_mode=fuel_pooling_mode,
            )
            if merged_tensor is None:
                print(f"🛑 Skipping timestamp {ts:04d} for keep_z_levels={keep_z_levels}.")
                continue

            print(f"\n💾 Saving merged tensor to '{output_file}'...")
            _raise_if_non_finite(
                merged_tensor,
                f"Merged tensor for timestamp {ts:04d}",
                output_file,
            )
            np.save(output_file, merged_tensor)
            print(f"🎉 Saved tensor for timestamp {ts:04d} with keep_z_levels={keep_z_levels}.")


def process_single_version(keep_z_levels=5, start_ts=286, end_ts=1999):
    """Backward-compatible wrapper for a single dataset version."""
    process_all_files(
        keep_z_levels_list=(keep_z_levels,),
        start_ts=start_ts,
        end_ts=end_ts,
        output_root="./final",
        data_root="./data",
    )


def build_argument_parser():
    """Build the command-line interface for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate fire-model tensors from ASC, flux, and fuel files."
    )
    parser.add_argument(
        "--keep-z-levels",
        nargs="+",
        type=int,
        default=[5],
        help="One or more z-depths to keep from the 46-level ASC file, e.g. --keep-z-levels 5 10 15.",
    )
    parser.add_argument(
        "--start-ts",
        type=int,
        default=150,
        help="First timestamp to process.",
    )
    parser.add_argument(
        "--end-ts",
        type=int,
        default=1424,
        help="Last timestamp to process.",
    )
    parser.add_argument(
        "--output-root",
        default=".",
        help="Directory where merged tensors will be written.",
    )
    parser.add_argument(
        "--data-root",
        default="../../tuo2-7_asc_1",
        help="Directory containing the FRFCA1M04ASC, flux, and fuel input files.",
    )
    parser.add_argument(
        "--fuel-pooling-mode",
        choices=("average", "max", "sum"),
        default="sum",
        help="Pooling method for the 360x360 fuel grid before downsampling.",
    )
    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    process_all_files(
        keep_z_levels_list=tuple(args.keep_z_levels),
        start_ts=args.start_ts,
        end_ts=args.end_ts,
        output_root=args.output_root,
        fuel_pooling_mode=args.fuel_pooling_mode,
        data_root=args.data_root,
    )


if __name__ == "__main__":
    main()