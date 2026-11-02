# Architecture

How the system is put together. Method details and motivation: [Edge_Detection_Research_Plan.md §8](../Edge_Detection_Research_Plan.md#8-our-method).

## 1. Overview

The system has two stages: an **offline teacher stage** that runs once, and a **training stage** that wraps the official PEdger training loop.

```
        once, offline                                   every training step (inside PEdger)
┌───────────────────────────────────┐      ┌──────────────────────────────────────────────────┐
│ SAUGE ─┐                          │      │ image, GT Y, teacher map T, agreement a          │
│ MuGE  ─┼─► mean ─► teacher map T  │─────►│   ─► same random crop/flip/rotate/scale          │
│ MEMO  ─┘   saved to disk          │      │   ─► PEdger forward (recurrent + non-recurrent)  │
└───────────────────────────────────┘      │   ─► soft target from Y, self-ensemble M and T,  │
                                           │      weighted per pixel by human agreement       │
                                           └──────────────────────────────────────────────────┘
Inference: unchanged PEdger (0.73M params, single network)
```

Key properties:

- Teachers never run during training, so training memory and speed stay close to PEdger's.
- Nothing is added to the network that is deployed. Inference cost is identical to PEdger.

## 2. Components

| Component | Location | Responsibility |
|---|---|---|
| Official PEdger | `third_party/PEdger/` | Networks, momentum copies, self-ensemble `M`, original loss. **Unmodified.** |
| Official PEdger++ | `third_party/PEdgerPlus/` | Only for the optional final experiment. |
| Data prep | `data/prepare_*.py` | Build train/val/test lists identical to PEdger's; MultiCue 3 seeded splits. |
| Agreement | `data/agreement.py` | `a(p)` → `agree(p)` → `α(p)`; optional 1–2 px dilation. |
| Teacher caching | `teachers/cache_*.py` | Run each teacher once on original training images; save 16-bit PNG probability maps. |
| Teacher ensemble | `teachers/ensemble.py` | Mean (default) or confidence-weighted ensemble `T`. |
| Teacher gate | `teachers/teacher_gate.py` | Score each teacher + ensemble vs PEdger on each validation split; drop teachers that are not better. |
| KD dataset | `pedger_kd/dataset.py` | Wraps PEdger's dataset; loads `T` and `a`; applies one shared geometric transform. |
| Variant A | `pedger_kd/soft_target.py` | Agreement-aware soft target (main method). |
| Variant B | `pedger_kd/kd_loss.py` | Weighted KD loss with logit standardization (simpler baseline). |
| Training loop | `pedger_kd/train_kd.py` | PEdger's loop with minimal hooks for A / B; AMP; per-epoch checkpoints; GPU-hour logging. |
| Evaluation | `eval/run_bsds_eval.py` | NMS + ODS / OIS / AP (MATLAB reference or validated Python port). |
| Agreement bins | `eval/agreement_bins.py` | Precision / recall per low / medium / high agreement bin (H2). |
| Benchmark | `bench/speed_flops.py` | Params, FLOPs (`fvcore` / `thop`), FPS (T4 + CPU), GPU-hours from logs. |
| Configs | `configs/*.yaml` | One file per experiment. |
| Notebooks | `notebooks/kaggle_*.ipynb` | One per stage; data and caches mounted as Kaggle Datasets. |

## 3. What PEdger does (the parts we hook into)

1. Two networks train together: a recurrent and a non-recurrent one.
2. Momentum copies are updated every epoch: `W_m ← (W_bp + W_m) / 2`.
3. Their predictions are fused per pixel by confidence (distance from 0.5) into **M**.
4. Soft target: `Ỹ = η_ep · M + (1 − η_ep) · Y`, with `η_ep = 0.8 · ep / EP`.
5. Training: 30 epochs, AdamW, 4-epoch warm-up to LR 0.001 then linear decay, batch 16 via gradient accumulation.
6. Inference: a single network.

**Variant A hooks step 4** (replaces the soft target). **Variant B hooks the loss** (adds a term after step 4). Nothing else changes.

## 4. Formulas

### Agreement (BSDS500, MultiCue)

`a(p) ∈ [0, 1]` = fraction of annotators marking pixel `p` (the averaged GT PNG).

```
agree(p) = |2·a(p) − 1|                       # 1 = all agree, 0 = evenly split
α(p)     = α_min + (1 − α_min) · agree(p)     # human-trust weight
```

### Teacher ensemble

```
default:   T(p) = mean_k T_k(p)
weighted:  T(p) = Σ_k c_k(p) · T_k(p) / Σ_k c_k(p),   c_k(p) = |2·T_k(p) − 1|
```

MuGE contributes the average of its granularity outputs.

### Variant A: target-level (main method)

```
Ỹ(p) = (1 − r·(1 − α(p))) · Y(p) + r·(1 − α(p)) · [ (1 − γ) · M(p) + γ · T(p) ]
```

- `r = ep / EP` is the optional warm-up ramp (`r = 1` disables it).
- `γ` is constant (0, 0.5, 1) or per-pixel: `γ(p) ∝ |2T(p) − 1| / (|2T(p) − 1| + |2M(p) − 1|)`.
- Sanity check: `γ = 0`, `r = 1`, `α(p) = 1 − η_ep` gives back PEdger exactly.

### Variant B: loss-level (baseline)

```
L = L_PEdger + λ_kd · Σ_p w(p) · BCE( σ(z_S(p)/τ), σ(z_T(p)/τ) ) / Σ_p w(p)
w(p) = 1 − α(p) + w_min
```

`z_S`, `z_T` are optionally z-scored per image before dividing by `τ` (logit standardization, adapted from [30]).

### Datasets without agreement (NYUDv2)

One annotation per image, so no `a(p)`. Two options:

- (a) constant `α = 1 − η_ep` (PEdger's schedule);
- (b) teacher confidence: `α(p) = 1 − c·|2T(p) − 1|`.

## 5. Data flow and storage

```
raw datasets ──prepare_*.py──► file lists (same as PEdger)
                       │
averaged BSDS/MultiCue GT ──agreement.py──► a(p) maps (on the fly or cached)
                       │
original train images ──cache_<teacher>.py──► teachers/<dataset>/<teacher>/*.png (16-bit)
                       │                          │
                       │                ensemble.py ──► T maps
                       │                teacher_gate.py ──► gate table (which teachers per dataset)
                       ▼
          pedger_kd/dataset.py: (image, Y, T, a) ──shared aug──► train_kd.py
                                                                   │
                                                 checkpoints/, logs (incl. GPU-hours)
                                                                   │
                                     eval/run_bsds_eval.py, eval/agreement_bins.py, bench/speed_flops.py
```

On Kaggle, raw datasets and teacher caches are stored as **Kaggle Datasets** and mounted read-only; checkpoints and logs go to the notebook's output.

## 6. Extension: PEdger++

PEdger++ uses the same soft-target form (`Ỹ = η·M + (1−η)·Y`), so Variant A's `soft_target` plugs in at the same hook. Use PEdger++'s "efficient" training to save GPU time. This is optional (hypothesis H5).

## 7. Invariants

- No teacher model, `T`, or `a(p)` is used at inference.
- Params / FLOPs / FPS of the trained student equal PEdger's.
- `T` and `a(p)` are always spatially aligned with the image after augmentation.
- With KD disabled (`lambda_kd = 0` / Variant A reduced to PEdger), training matches the baseline.
