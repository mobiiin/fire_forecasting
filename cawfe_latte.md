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

## Full CAWFE-Latte

`cawfe_latte` is the full paper architecture: CAWFE-Latte: Layer-Aware Temporal Transformer Neural Operator for Energy Release and Fire Spread Forecasting.

CAWFE-Latte is a fire-aware spatiotemporal neural operator that explicitly separates atmospheric drivers from fire/fuel state variables, encodes vertical atmospheric structure, uses wind-guided and fire-front-guided feature modulation, combines local attention with long-range Mamba-style transport, and applies neural-operator bottleneck mixing for field-to-field prediction of fuel consumption, fire perimeter, and energy release.

CAWFE-Latte-Lite implements the core modular architecture. Full CAWFE-Latte adds two physical inductive biases:

- wind-guided directional modulation for wind-driven spread
- AFNO-style neural-operator bottleneck mixing for global field-to-field coupling

Architecture flow:

```text
CAWFE sequence
↓
Vertical Atmospheric Token Encoder
↓
Fire/Fuel State Encoder
↓
Wind-Guided Directional Module
↓
Fire-Front Attention Gate
↓
Hybrid Transformer + Mamba Backbone
↓
Neural Operator Bottleneck
↓
Multi-Task Fire Decoder
↓
Physical Output Constraints
```

The Wind-Guided Directional Module extracts low-level `U/V` wind from raw atmospheric channels when engineered wind channels are not configured. It computes wind speed, `cos`, and `sin`, then modulates fused feature maps according to wind-driven spread tendency.

The Neural Operator Bottleneck treats CAWFE prediction as a field-operator learning problem. The first implementation uses AFNO-style spectral mixing: bottleneck feature maps are transformed with FFTs, mixed in Fourier space, shrunk for sparsity, inverted, and passed through a pointwise MLP. This adds global coupling beyond local convolution and attention.

The decoder predicts the same four dense maps as Latte-Lite:

- surface fuel consumption
- canopy fuel consumption
- fire/perimeter mask logits
- `log1p` energy release map

Optional physical constraints apply `softplus` to consumed fuel and energy release while preserving mask logits.

## CAWFE-Latte vs CAWFE-Latte-Lite

| Module | Latte-Lite | Full Latte |
| --- | --- | --- |
| vertical atmosphere encoder | yes | yes |
| fire/fuel encoder | yes | yes |
| fire-front gate | yes | yes |
| hybrid transformer + mamba | yes | yes |
| wind-guided directional module | no | yes |
| neural operator bottleneck | no | yes |
| physical constraints | optional | optional |

## Full Model Ablation Plan

Full CAWFE-Latte comparisons:

- ConvLSTM U-Net
- Earthformer-lite
- WeatherFormer-lite
- CAWFE-ST-Mamba
- CAWFE-Latte-Lite
- CAWFE-Latte full
- full without wind guidance
- full without neural operator
- full without fire-front gate
- full without vertical encoder
- full transformer-only
- full mamba-only

Generate full-model ablation configs:

```bash
python scripts/ablate_cawfe_latte.py --base_config configs/default.yaml --output_dir configs/ablations/cawfe_latte/
```

## Full Model Commands

Smoke test:

```bash
python scripts/smoke_test_cawfe_latte.py --config configs/default.yaml
```

Train:

```bash
python scripts/train_cawfe_latte.py --config configs/default.yaml
```

Test:

```bash
python scripts/test_cawfe_latte.py --config configs/default.yaml --checkpoint artifacts/checkpoints/cawfe_latte/best_model.pt --split test
```

Aux visualization:

```bash
python scripts/visualize_cawfe_latte_aux.py --config configs/default.yaml --checkpoint artifacts/checkpoints/cawfe_latte/best_model.pt --split test --num_samples 5
```
