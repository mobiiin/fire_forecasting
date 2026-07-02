# `bigtensor.py`

`bigtensor.py` builds one merged NumPy tensor per timestamp from:

- the main ASC file
- the `.flux` file
- the `.fuel` file

The script is now mostly automatic. In the normal case you only need to point it at the dataset folder.

## What is auto-detected

From `--data-root`, the script now detects:

- the dataset file stem, such as `KINGNSM04ASC` or `CALWD1M04ASC`
- the available timestamp range by scanning the `####` number at the end of filenames
- the flux shape from the first line of the first flux file
- the fuel shape from the first line of the first fuel file
- the ASC shape from the flux shape as `(flux_x, flux_y, 46, 10)`

It also prints the detected dataset start and end timestamp in the terminal before processing.

## Filename pattern

The script expects files like:

- `CALWD1M04ASC.0120`
- `CALWD1M04ASC.flux.0120`
- `CALWD1M04ASC.fuel.0120`

The dataset stem is everything before the optional `.flux` or `.fuel` part and before the final 4-digit timestamp.

If the folder contains only one dataset stem, the script selects it automatically.

If the folder contains multiple stems, pass `--file-stem` explicitly.

## Shape detection

The script reads the first line of the flux file and fuel file.

Example flux header:

```text
216 168 4 484.524 370.370
```

This gives:

- flux shape: `(216, 168, 4)`
- ASC shape: `(216, 168, 46, 10)`

Example fuel header:

```text
1080 840 2 96.905 74.074
```

This gives:

- fuel shape: `(1080, 840, 2)`

## Defaults

- `keep_z_levels` defaults to `8`
- `pool_window` defaults to `(5, 5)`
- `start_ts` and `end_ts` default to auto-detect
- `file_stem` defaults to auto-detect
- `asc_shape`, `flux_shape`, and `fuel_shape` default to auto-detect

## Simplest usage

```bash
python3 bigtensor.py \
  --data-root ../data \
  --output-root tensors
```

That is enough when the folder contains one dataset stem and you want the full detected timestamp range with the default `keep_z_levels=8`.

## Common examples

Process the full dataset with default `keep_z_levels=8`:

```bash
python3 bigtensor.py \
  --data-root ../calwd1_data \
  --output-root tensors_calwd1
```

Process only part of the detected range:

```bash
python3 bigtensor.py \
  --data-root ../calwd1_data \
  --start-ts 120 \
  --end-ts 240 \
  --output-root tensors_calwd1
```

Process multiple retained z-level settings:

```bash
python3 bigtensor.py \
  --data-root ../calwd1_data \
  --keep-z-levels 5 8 10 \
  --output-root tensors_calwd1
```

If a folder contains multiple dataset stems:

```bash
python3 bigtensor.py \
  --data-root ../mixed_data \
  --file-stem CALWD1M04ASC \
  --output-root tensors_calwd1
```

If you need to override the detected shapes manually:

```bash
python3 bigtensor.py \
  --data-root ../calwd1_data \
  --asc-shape 216 168 46 10 \
  --flux-shape 216 168 4 \
  --fuel-shape 1080 840 2 \
  --output-root tensors_calwd1
```

## Notes

- The script processes timestamps only when all three files exist for the same timestamp: ASC, flux, and fuel.
- Flux and ASC must share the same `x, y` grid.
- Fuel must be divisible by the pooling window.
- After pooling, fuel must match the ASC/flux `x, y` grid.
- Header skipping is unchanged:
  - ASC skips `0` rows
  - flux skips `1` row
  - fuel skips `1` row

## Output shape

The merged tensor shape is:

```text
(asc_x, asc_y, keep_z_levels * asc_fields + flux_channels + fuel_channels)
```

With the standard dataset layout that becomes:

```text
(x, y, keep_z_levels * 10 + 6)
```
