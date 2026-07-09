# Run Guide

Project base directory: `/home/mhabibp/fire_forecasting`

Before running anything else, verify that `configs/default.yaml` is configured for your manual fire split:

- `split_mode: manual_fire_holdout`
- `manual_fire_split.train_fires`, `val_fires`, and `test_fires` contain your hand-picked fires
- every listed fire exists in `fire_dataset_index.json`
- `main_data_dir` points to `/media/mhabibp/Elements/Mobin_CPS_files/New_CAWFE/`

If you want a fresh dataset index, regenerate it first:

```bash
python scripts/discover_fire_datasets.py --main_data_dir /media/mhabibp/Elements/Mobin_CPS_files/New_CAWFE/
```

That command should rewrite `fire_dataset_index.json` in the project root. If any fire path is stale or missing, update the manual fire lists in `configs/default.yaml` before continuing.

Then run the intermediate validation step:

```bash
python scripts/compute_normalization.py --config configs/default.yaml
```

This should:

- load only the training fires
- save normalization stats to `artifacts/normalization_stats.npz`
- save the resolved split file to `artifacts/splits/manual_fire_split_resolved.json`

Optional inspection steps before training:

```bash
python scripts/inspect_dataset.py --config configs/default.yaml
python scripts/inspect_target_channels.py --config configs/default.yaml
python scripts/inspect_energy_release.py --config configs/default.yaml --split train
```

Use these checks to confirm:

- train, validation, and test fire names and sample counts look right
- target channels are correct
- energy-release geometry is sane

Train the model:

```bash
python scripts/train_convlstm_unet.py --config configs/default.yaml
```

Expected outputs:

- checkpoints under `artifacts/checkpoints/`
- logs under `artifacts/logs/`
- training and validation loss printed each epoch

Test the trained model:

```bash
python scripts/test_model.py --config configs/default.yaml
```

This evaluates the model on the held-out test fires from your manual split.

Visualize results:

```bash
python scripts/visualize_predictions.py --config configs/default.yaml
```

If your branch supports a split flag, you can also run:

```bash
python scripts/visualize_predictions.py --config configs/default.yaml --split test
```

Shortest end-to-end command sequence:

```bash
python scripts/discover_fire_datasets.py --main_data_dir /media/mhabibp/Elements/Mobin_CPS_files/New_CAWFE/
python scripts/compute_normalization.py --config configs/default.yaml
python scripts/train_convlstm_unet.py --config configs/default.yaml
python scripts/test_model.py --config configs/default.yaml
python scripts/visualize_predictions.py --config configs/default.yaml
```

After the run, inspect these outputs:

- `fire_dataset_index.json`
- `artifacts/splits/manual_fire_split_resolved.json`
- `artifacts/normalization_stats.npz`
- `artifacts/checkpoints/`
- `artifacts/logs/`
- the visualization output directory configured for your run

If training fails with a data-directory error, rerun dataset discovery so `fire_dataset_index.json` matches the mounted `New_CAWFE` tree, then fix the manual fire lists if needed.