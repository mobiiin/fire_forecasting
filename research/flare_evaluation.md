# FLARE paper evaluation utilities

Run from the repository root in the `fire_forecasting` conda environment. These tools use the existing CAWFE-Latte architecture and frozen checkpoints; the paper labels are `FLARE baseline` and `FLARE final`.

## Locked Table 2 protocol

The source is the processed `sparse5_h10` temporal index at `/scratch/mhabibp/cawfe_datasets/cawfe_engineered_v1/indices/temporal/samples_sparse5_h10.jsonl`. The test partition contains 59,509 deterministic 64 × 64 patches, with five sparse input frames and a ten-frame forecast horizon. It includes all activity levels, including no-fire patches. The seven held-out fire IDs are `CHIMNEYTOPS2`, `MCKINNEY`, `RIM__0822__keepz_08`, `SPRINGS`, `THOMAS__1210__keepz_08`, `TUOLUMNE_65`, and `WOOLSEY`.

Table 2 used `src.evaluation.held_out_test.evaluate_locked_test` with `FullValidationAccumulator`. Each patch contributes its pixels to global per-fire counts and regression sums. Dice is computed from the summed true-positive, false-positive, and false-negative counts, not a mean of patch Dice. Overlapping sliding patches are **not** reconstructed into full-domain frames before scoring. Target fire pixels are `fire_mask > 0.5`; predicted fire pixels are `sigmoid(mask_logit) > 0.5`. Active Canopy MAE uses target-active pixels only. The four output channels are surface consumption, canopy consumption, fire-mask logit, and `log1p` energy in MW. The normalizer comes from the training-only `latest_normalization_sparse5_h10.json` file recorded in each resolved config. See `src/evaluation/full_validation.py` and `src/data/processed_sample_dataset.py` for the actual definitions and target loading.

All three learned models have seeds 42, 123, and 2026. The selected `best_model.pt` paths are below. Each checkpoint's configuration is `resolved_config.yaml` in that run directory, or `configs/resolved_config.yaml` when the top-level copy is absent. Each learned run's normalization path is `/scratch/mhabibp/cawfe_datasets/cawfe_engineered_v1/normalization/latest_normalization_sparse5_h10.json`. The utility resolves this path from each run's config and fails if the file is missing.

| Paper model | Seed 42 | Seed 123 | Seed 2026 |
|---|---|---|---|
| CAWFE-ST-Mamba | `artifacts/table2_baselines/cawfe_st_mamba/seed_42/slurm16186252/checkpoints/best_model.pt` | `artifacts/table2_baselines/cawfe_st_mamba/seed_123/slurm16186253/checkpoints/best_model.pt` | `artifacts/table2_baselines/cawfe_st_mamba/seed_2026/slurm16186254/checkpoints/best_model.pt` |
| FLARE baseline | `artifacts/final_training/cawfe_latte/baseline/seed_42/slurm16085942/checkpoints/best_model.pt` | `artifacts/final_training/cawfe_latte/baseline/seed_123/slurm16105381/checkpoints/best_model.pt` | `artifacts/final_training/cawfe_latte/baseline/seed_2026/slurm16105382/checkpoints/best_model.pt` |
| FLARE final | `artifacts/final_training/cawfe_latte/GA_Q2/seed_42/slurm16085948/checkpoints/best_model.pt` | `artifacts/final_training/cawfe_latte/GA_Q2/seed_123/slurm16085949/checkpoints/best_model.pt` | `artifacts/final_training/cawfe_latte/GA_Q2/seed_2026/slurm16105386/checkpoints/best_model.pt` |

Persistence is deterministic and has no checkpoint. Its canonical run is `artifacts/table2_baselines/persistence`, using `ProcessedHistoryBaselinePredictor` and only observed input frames.

## Per-fire table

```bash
python scripts/evaluate_per_fire.py --models persistence,cawfe_st_mamba,cawfe_latte_baseline,cawfe_latte_final --seeds 42,123,2026
python scripts/evaluate_per_fire.py --models cawfe_latte_final --seeds 42
```

By default the script validates and republishes the complete locked Table 2 per-fire results already stored with each run. This avoids re-running 59,509 patch inferences per seed. Pass `--recompute` to run the same canonical evaluator again on CPU and save the new locked evaluation under `results/per_fire_generalization/recomputed/`. The CPU Mamba path uses `mamba_ssm`'s own PyTorch reference selective scan because its fused kernels require CUDA; it keeps the trained weights and architecture. Full CPU recomputation, especially ST-Mamba, can take a long time.

Outputs are `results/per_fire_generalization/per_fire_raw.csv`, `per_fire_summary.csv`, `per_fire_summary.txt`, `per_fire_table.tex`, and `provenance.txt`. The CSV contains all six main metrics plus the optional Table 2 metrics. Standard deviations are sample standard deviations across available seeds; persistence has zero standard deviation.

CPU SLURM examples:

```bash
MODELS="persistence,cawfe_st_mamba,cawfe_latte_baseline,cawfe_latte_final" SEEDS="42,123,2026" sbatch slurm/eval_per_fire_cpu.slurm
MODELS="cawfe_latte_final" SEEDS="42" sbatch slurm/eval_per_fire_cpu.slurm
```

Set `RECOMPUTE=1` to force CPU inference. The script uses the repository's existing `work1` partition, account, and conda setup; it requests no GPU.

## Qualitative case workflow

For four-column comparison PNGs, run `python scripts/qualitative_test_comparison.py CHIMNEYTOPS2 10`. The two arguments are a fire name or `test`, `val`, `train`, or `all`, and the number of PNG figures per fire. This is the direct figure workflow; the exporter below serves separate model-panel diagnostics.

The dataset stores frame indices, not wall-clock timestamps. `--timestamp` accepts an exact current or target frame index; `--sample-id` resolves a specific spatial patch. The case manifest stores both, the full patch coordinates, split, activity statistics, and SHA-256 hashes of the raw inputs, target, and terrain. Every model checks those hashes against the held-out dataset before inference. Selection modes use only the target or index metadata. `high_activity` and `low_activity` rank positive-fire patches by ground-truth fire-mask fraction.

```bash
python scripts/export_qualitative_forecasts.py --fire CHIMNEYTOPS2 --list-fire-samples
python scripts/export_qualitative_forecasts.py --fire CHIMNEYTOPS2 --num-samples 3 --selection-mode high_activity --selection-seed 123 --save-case-manifest qualitative_cases/chimney_cases.json
python scripts/export_qualitative_forecasts.py --model persistence --case-manifest qualitative_cases/chimney_cases.json --save-targets --save-predictions --save-arrays --save-png --save-metrics --save-metadata
python scripts/export_qualitative_forecasts.py --model cawfe_st_mamba --checkpoint-seed 42 --case-manifest qualitative_cases/chimney_cases.json --save-targets --save-predictions --save-arrays --save-png --save-metrics --save-metadata
python scripts/export_qualitative_forecasts.py --model cawfe_latte_baseline --checkpoint-seed 42 --case-manifest qualitative_cases/chimney_cases.json --save-targets --save-predictions --save-arrays --save-png --save-metrics --save-metadata
python scripts/export_qualitative_forecasts.py --model cawfe_latte_final --checkpoint-seed 42 --case-manifest qualitative_cases/chimney_cases.json --save-targets --save-predictions --save-arrays --save-png --save-metrics --save-metadata
```

`--output-dir` defaults to `qualitative_results`. Each case gets a fire/case folder with `manifest.json`, a `target/` folder, and one separate model/seed folder. `prediction.npz` contains the four raw output channels, raw mask logits, sigmoid probabilities, and the thresholded mask. `target.npz` contains the canonical target channels. `--save-inputs` saves raw inputs plus the model input. PNG and PDF panels are independent files. The energy colorbar is labeled `log1p energy (MW)`. `--scale-mode ground_truth` is the default and fixes each channel's limits from the target, so sequential model runs share a scale. `--scale-mode shared_case` includes prediction arrays already present in that case folder when choosing limits; regenerate earlier model panels with `--overwrite` after adding models if you need that expanded scale on every existing panel. `--vmin` and `--vmax` override computed limits.

`--context latest_fire,low_level_wind,terrain` exports the observed-interval fire mask, low-level mean wind speed with U/V vectors in NPZ, and terrain arrays when available. The wind channels follow the raw-frame layout and low-level indices in the resolved config. `--overwrite` is required to replace an existing model's case folder. The qualitative exporter accepts `--device cpu` or `--device cuda`.
