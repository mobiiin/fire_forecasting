# ConvLSTM U-Net Wildfire Forecasting

## Project Overview
This project trains a spatiotemporal deep learning model to predict future wildfire intensity or perimeter maps from multimodal gridded simulation outputs. The core model is ConvLSTM U-Net: ConvLSTM encodes temporal fire evolution, and U-Net decodes dense 2D top-view predictions.

## Dataset Format
- One `.npy` file per timestamp.
- Each tensor is shaped `(144, 144, C)`.
- Files are sorted chronologically before split/training.
- Timestamps are approximately one minute apart.
- Example channel count: `C = 56` for 5 vertical levels (`5 * 10` atmospheric + `4` flux + `2` fuel channels).
- Input sequence for one sample:
	- `X` raw shape: `(T_in, 144, 144, C)`
	- `X` DataLoader shape: `(B, T_in, C, 144, 144)`
- Target shape:
	- `y` DataLoader shape: `(B, 1, 144, 144)`
- Splits must be chronological, not random, to prevent future information leakage into earlier training windows.

## Model Overview
- ConvLSTM learns temporal wildfire dynamics from the input sequence.
- U-Net generates a dense spatial output map.
- Output is a top-view 2D future map:
	- continuous fire intensity (regression), or
	- binary fire perimeter/fire mask (segmentation).

## Repository Structure
Expected project layout:

```text
fire_forecasting/
	configs/
	src/
	scripts/
	outputs/
	data/
```

## Conda Environment Setup
Create and activate the environment before running any scripts:

```bash
conda create -n fire_forecasting python=3.10 -y
conda activate fire_forecasting
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## GPU / CUDA Note
PyTorch installation may need to be adjusted for your CUDA version.

Quick check:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If CUDA is not available but you expected GPU usage, install a CUDA-compatible PyTorch build for your system.

## Configuration
Main config file: `configs/default.yaml`

Important fields:
- `data_dir`
- `file_pattern`
- `input_sequence_length`
- `prediction_horizon`
- `target_channel`
- `task_type`
- `fire_threshold`
- `batch_size`
- `learning_rate`
- `epochs`
- `checkpoint_dir` (mapped in this project as `checkpoint.path` parent directory)
- `normalization_stats_path` (mapped in this project as `normalization.path`)

## Common Commands
Environment:

```bash
conda create -n fire_forecasting python=3.10 -y
conda activate fire_forecasting
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Inspect data:

```bash
python scripts/inspect_dataset.py --config configs/default.yaml
```

Inspect selected target channels:

```bash
python scripts/inspect_target_channels.py --config configs/default.yaml --channels 50 51 52 53 54 55 --timesteps 0 100 500 1000 2000 3000
```

Compute normalization:

```bash
python scripts/compute_normalization.py --config configs/default.yaml
```

If `normalization.normalize_target: true`, rerun this after config or dataset changes so the archive includes target-channel stats as well as input-channel stats.

Sanity check:

```bash
python scripts/sanity_check_project.py --config configs/default.yaml
```

Quick smoke test:

```bash
bash scripts/run_quick_smoke_test.sh configs/default.yaml
```

Full pipeline:

```bash
bash scripts/run_full_pipeline.sh configs/default.yaml
```

Train:

```bash
python scripts/train_convlstm_unet.py --config configs/default.yaml
```

Test:

```bash
python scripts/test_model.py --config configs/default.yaml
```

Evaluate persistence baseline:

```bash
python scripts/evaluate_persistence_baseline.py --config configs/default.yaml
```

Visualize:

```bash
python scripts/visualize_predictions.py --config configs/default.yaml --num_samples 10
```

Rollout:

```bash
python scripts/rollout_predictions.py --config configs/default.yaml --start_index 0 --rollout_steps 30
```

If shell scripts are not executable yet:

```bash
chmod +x scripts/*.sh
```


## End-to-End Workflow
Run these commands in order from project root:

1. Inspect dataset

```bash
python scripts/inspect_dataset.py --config configs/default.yaml
```

2. Compute normalization stats

```bash
python scripts/compute_normalization.py --config configs/default.yaml
```

3. Run project sanity checks

```bash
python scripts/sanity_check_project.py --config configs/default.yaml
```

4. Train model

```bash
python scripts/train_convlstm_unet.py --config configs/default.yaml
```

5. Visualize predictions

```bash
python scripts/visualize_predictions.py --config configs/default.yaml --num_samples 10
```

6. Evaluate on test split

```bash
python scripts/test_model.py --config configs/default.yaml
```

7. Run rollout prediction

```bash
python scripts/rollout_predictions.py --config configs/default.yaml --start_index 0 --rollout_steps 30
```

## Training Instructions
- Start with regression mode first.
- Recommended initial settings:
	- `input_sequence_length: 10`
	- `prediction_horizon: 1`
	- `task_type: regression`
	- `loss_type: huber`
	- `batch_size: 1` or `2` if GPU memory is limited
- Use chronological split only.
- Monitor validation loss and inspect saved visualizations frequently.

## Perimeter Segmentation Mode
- Set `task_type: segmentation`.
- Set `fire_threshold` to convert intensity to perimeter mask.
- Use BCE + Dice (`loss_type: bce_dice`).
- Model output should remain logits.
- Visualization applies sigmoid and thresholding for perimeter display.

## Visualization
Prediction visualization outputs include:
- input/current fire intensity map
- ground truth future map
- predicted future map
- absolute error map
- optional contour overlay
- rollout animation (GIF/MP4 based on available writer)

## Troubleshooting
- No files found
	- Verify `data_dir` and `file_pattern` in config.
	- Confirm you run commands from project root.

- Wrong file sorting
	- Ensure filenames contain increasing numeric suffixes or sortable timestamp names.
	- Use `inspect_dataset.py` to verify first/last filenames are chronologically ordered.

- Shape mismatch
	- Required model input: `(B, T, C, H, W)`.
	- Required target shape: `(B, 1, H, W)`.
	- Confirm channel count and patch settings (`use_patches`, `patch_size`).

- Invalid target channel
	- `target_channel` must be in `[0, C-1]`.
	- Run inspect/sanity check scripts to confirm tensor `C`.

- NaNs or infs
	- Run dataset inspection and check NaN/Inf counts.
	- Remove or repair problematic tensors before training.

- CUDA out of memory
	- Lower `batch_size`.
	- Reduce `input_sequence_length`.
	- Enable patch training and reduce `patch_size`.

- Poor predictions due to class imbalance
	- Use segmentation mode with `bce_dice`.
	- Increase active-focused patch sampling (`active_patch_probability`).
	- Tune `fire_threshold` and `active_threshold`.

- Data leakage from random splitting
	- Do not random split timestamps.
	- Keep chronological split logic unchanged.

- Normalization stats missing
	- Run:

	```bash
	python scripts/compute_normalization.py --config configs/default.yaml
	```

## Suggested Experiment Ladder
1. Persistence baseline
2. Current-frame U-Net
3. ConvLSTM U-Net
4. ConvGRU U-Net (if implemented later)
5. Attention ConvLSTM U-Net (later)
6. Patch-based training (later)
