# Fire Simulation Tensor Dataset

This workspace contains time-series output from a fire simulation model. Each timestamp is stored in several text files, and the goal of `bigtensor.py` is to convert those files into NumPy tensors for machine learning.

## What the data represents

The data is a sequence of model snapshots. The filename suffix, such as `0287`, means the output number after model start. In this dataset, the outputs are one minute apart, so the suffix is effectively the number of minutes after simulation start.

At each timestamp, the simulation writes multiple files:

- `KINGNSM04ASC.xxxx` - main atmospheric 3D fields
- `KINGNSM04ASC.flux.xxxx` - surface and canopy flux fields
- `KINGNSM04ASC.fuel.xxxx` - remaining fuel on the refined fire grid

## Header-row handling

The text files do not all start the same way, so the Python loader treats them differently:

- `KINGNSM04ASC.xxxx` - no header row is skipped
- `KINGNSM04ASC.flux.xxxx` - the first row is skipped
- `KINGNSM04ASC.fuel.xxxx` - the first row is skipped

This is already built into `bigtensor.py`.

## Main atmospheric file: `KINGNSM04ASC.xxxx`

The main `.ASC` file contains 3D atmospheric variables. The first line in the corresponding header file begins with `10`, which means there are 10 3D fields in the file.

The 10 variables are:

1. U compon - west/east wind component, m/s
2. V compon - south/north wind component, m/s
3. W compon - vertical wind velocity, m/s
4. POT TEMP - potential temperature, K
5. PRES PRT - pressure perturbation, mb
6. BUOYANCY - buoyancy, K
7. EDDY DIF - eddy diffusion coefficient, m**2/s
8. VAPOR MR - vapor mixing ratio, g/kg
9. CLOUD MR - cloud mixing ratio, g/kg
10. SMOKE MR - smoke mixing ratio, g/kg

## Meaning of 46

The `46` used in the code is the number of vertical grid cells in the model, also called the Z dimension or vertical levels.

So when the script reshapes the atmospheric file, it treats one timestamp as:

```text
(144, 144, 46, 10)
```

That means:

- 144 cells in X direction
- 144 cells in Y direction
- 46 cells in Z direction
- 10 atmospheric variables
The unpacking follows the same explicit channel-split style used for the flux and fuel files:

- the flat ASC list is divided into 46 consecutive z-level blocks
- each z-level block is divided into 10 consecutive field blocks
- each field block is reshaped into a 144 x 144 matrix with `order="C"`

In other words, each of the 10 variables has values on a 144 x 144 x 46 3D grid, but the raw list is unpacked as 46 z-level groups of 10 two-dimensional matrices.

## Flux file: `KINGNSM04ASC.flux.xxxx`

The `.flux` file contains 4 2D fields on the 144 x 144 atmospheric grid.

The 4 flux variables are:

1. surface sensible heat flux
2. surface latent heat flux
3. canopy sensible heat flux
4. canopy latent heat flux

The file header line begins with:

```text
144 144 4 ...
```

That means:

- 144 x cells
- 144 y cells
- 4 variables

These fluxes are typically combined or stacked as 4 channels in the merged tensor.

## Fuel file: `KINGNSM04ASC.fuel.xxxx`

The `.fuel` file contains fuel remaining on the refined fire grid.

Its grid size is:

```text
720 720 2 ...
```

That means:

- 720 x cells
- 720 y cells
- 2 variables

The 2 variables are:

1. surface fuel load
2. canopy fuel load

Because the fuel grid is 5x finer than the atmospheric grid, the fuel data is downsampled from 720 x 720 to 144 x 144 using 5 x 5 average pooling.

## Recommended merged tensor format

For machine learning, the merged per-timestamp representation keeps all 10 ASC variables for each retained z-level and then appends flux and fuel channels.

```text
(144, 144, keep_z_levels * 10 + 4 + 2)
```

For example, if you keep 5 z-levels, the tensor shape is:

```text
(144, 144, 56)
```

Where the channels are:

- 10 atmospheric features per retained z-level
- 4 flux features
- 2 fuel features

This layout is practical if the target is fire intensity or fire perimeter prediction, because each grid cell gets a feature vector containing the full vertical atmospheric slice that you chose to keep, plus flux and fuel information.

## Why this layout is useful

This layout keeps all features aligned by spatial location:

- atmospheric variables describe local fire/weather state
- flux variables describe surface energy exchange
- fuel variables describe available burnable material

That means each cell in the 144 x 144 grid becomes one supervised learning example with `keep_z_levels * 10 + 6` input features, and the model can learn both local and spatially coupled behavior.

## Suggested ML usage

For time-series learning, the usual next step is to stack multiple timestamps into sequences:

```text
(T, 144, 144, keep_z_levels * 10 + 6)
```

Where `T` is the number of timesteps in the input window.

Typical training setups are:

- predict fire intensity at the next timestamp
- predict a fire perimeter mask at the next timestamp
- predict several future timestamps ahead

## Notes about the current Python script

The file `bigtensor.py` currently does the following:

- reads the main atmospheric file and reshapes it into a 4D tensor
- loads flux data and pads it to match the atmospheric vertical size
- pools the fuel file from 720 x 720 down to 144 x 144
- concatenates the results into a larger tensor and saves it as `.npy`

## Usage

Run the script from this directory with one or more z-level values:

```bash
python3 bigtensor.py --keep-z-levels 5
```

To generate multiple dataset versions in one run:

```bash
python3 bigtensor.py --keep-z-levels 5 10 15
```

You can also limit the timestamps and change the output directory:

```bash
python3 bigtensor.py --keep-z-levels 5 --start-ts 287 --end-ts 300 --output-root tensors --fuel-pooling-mode sum
```

The script writes one `.npy` file per timestamp under a versioned folder such as `tensors/keepz_05/`.

Important clarification:

- `46` is the Z dimension of the atmospheric grid
- the 10 atmospheric features are repeated across the 46 vertical levels in the raw 3D data
- flux and fuel are 2D grid fields, so they must be handled differently from the atmospheric volume

## Good next step for machine learning

If the goal is ML on fire dynamics, the best long-term approach is to build one merged tensor per timestamp with shape `144 x 144 x (keep_z_levels * 10 + 6)`, then assemble those tensors into a time sequence and split the sequence by time, not randomly.

That avoids leakage between train and test data and keeps the forecast task realistic.

## Example channel order

A consistent channel order should be documented and reused everywhere.

One reasonable order is:

```text
U, V, W, POT TEMP, PRES PRT, BUOYANCY, EDDY DIF, VAPOR MR, CLOUD MR, SMOKE MR,
flux_sensible_surface, flux_latent_surface, flux_sensible_canopy, flux_latent_canopy,
fuel_surface, fuel_canopy
```

## Summary

- `46` is the number of vertical levels in the atmospheric model grid
- the atmospheric file contains 10 3D fields
- the flux file contains 4 2D fields
- the fuel file contains 2 2D fields on a 720 x 720 refined grid
- the merged tensor keeps all 10 atmospheric variables for each retained z-level
- for sequence models, stack those tensors across time into `(T, 144, 144, keep_z_levels * 10 + 6)`
