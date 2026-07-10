# CAWFE-Latte

CAWFE-Latte is the custom paper architecture for this project. CAWFE data are not generic image tensors: the input combines vertical atmospheric profiles, surface/canopy fire state, fuel and flux variables, temporal fire evolution, and dense spatial targets around a moving fire front. CAWFE-Latte-Lite is the first implemented version of this architecture.

## Input And Output

Canonical input:

```text
X: (B, 6, 129, 64, 64)
```

Canonical output:

```text
pred: (B, 4, 64, 64)
```

Output channels:

- `pred[:, 0]`: surface consumed fuel
- `pred[:, 1]`: canopy consumed fuel
- `pred[:, 2]`: fire/perimeter mask logits
- `pred[:, 3]`: `log1p` energy release total MW

The model does not apply sigmoid to the mask channel.

## CAWFE-Latte-Lite Architecture

`cawfe_latte_lite` is implemented as a modular sequence-to-map model:

1. Vertical Atmospheric Token Encoder
2. Fire/Fuel State Encoder
3. Feature Fusion
4. Fire-Front Attention Gate
5. Hybrid Transformer + Mamba Backbone
6. Multi-Task Fire Decoder
7. Optional Physical Output Constraints

The vertical encoder reshapes raw atmospheric channels `0-79` into `8` vertical levels by `10` variables, then encodes level-wise structure with attention, MLP, Conv1d, or flat-conv ablation modes.

The fire/fuel encoder separately processes flux channels `80-83`, fuel channels `84-85`, and engineered channels `86-128`. This keeps fire state variables distinct from atmospheric drivers before fusion.

The fire-front gate learns a per-time, per-pixel attention map from fire/fuel/engineered channels and uses it to emphasize likely active-front regions in the fused features.

The hybrid backbone uses local spatial window attention for combustion and perimeter detail, plus tri-axis Mamba-style sequence mixing over time, width, and height for longer-range temporal/spatial transport. It supports `transformer_only`, `mamba_only`, `hybrid_transformer_mamba`, and `conv_only` ablations.

The decoder performs U-Net-style dense prediction from the high-resolution skip and lower-resolution bottleneck. Separate task heads are the default for surface fuel, canopy fuel, mask logits, and energy release.

Optional physical constraints apply `softplus` to consumed fuel and energy outputs while leaving the mask channel as logits.

## Ablation Plan

Core comparisons:

- ConvLSTM U-Net
- CAWFE-Latte-Lite full
- no vertical encoder
- no fire/fuel encoder
- no fire-front gate
- transformer-only backbone
- mamba-only backbone
- no physical constraints
- conv-only backbone

Generate ablation configs:

```bash
python scripts/ablate_cawfe_latte_lite.py --base_config configs/default.yaml --output_dir configs/ablations/cawfe_latte_lite/
```

## Commands

Smoke test:

```bash
python scripts/smoke_test_cawfe_latte_lite.py --config configs/default.yaml
```

Train:

```bash
python scripts/train_cawfe_latte_lite.py --config configs/default.yaml
```

Test:

```bash
python scripts/test_cawfe_latte_lite.py --config configs/default.yaml --checkpoint artifacts/checkpoints/cawfe_latte_lite/best_model.pt --split test
```

## Future Full CAWFE-Latte

The full CAWFE-Latte version will add a neural-operator bottleneck, stronger wind-guided directional modules, and potentially a multi-step sequence decoder. CAWFE-Latte-Lite keeps those hooks out of the main path for now while preserving the ablation surface needed for paper experiments.
