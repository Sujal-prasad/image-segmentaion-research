# Implementation

Module-by-module guide for building the code. Design: [architecture.md](architecture.md). Rules: [AGENTS.md](AGENTS.md). Source plan: [§10 What We Have to Code](../Edge_Detection_Research_Plan.md#10-what-we-have-to-code).

Environment: Kaggle T4 (16 GB), PyTorch with `torch.cuda.amp`, checkpoint every epoch. Build on the official PEdger code; do not rewrite it.

## Repository layout

```
pedger-mtkd/
├── third_party/
│   ├── PEdger/                  # official PEdger code (unmodified submodule)
│   └── PEdgerPlus/              # official PEdger++ code (optional final experiment)
├── configs/                     # YAML per experiment
├── data/
│   ├── prepare_bsds.py          # BSDS-aug + PASCAL lists (same as PEdger)
│   ├── prepare_nyud.py
│   ├── prepare_multicue.py      # 3 random splits, fixed seeds
│   └── agreement.py             # a(p), agree(p), α(p) from averaged GT (± dilation)
├── teachers/
│   ├── cache_sauge.py           # run each teacher ONCE on original training images
│   ├── cache_muge.py            #   → save probability maps as 16-bit PNG
│   ├── cache_memo.py            #   (cache_diffusionedge.py if MEMO weights are unavailable)
│   ├── ensemble.py              # mean / confidence-weighted teacher map T
│   └── teacher_gate.py          # score each teacher + ensemble vs PEdger on each validation split
├── pedger_kd/
│   ├── dataset.py               # wraps PEdger dataset; loads T and a(p); SAME geometric augmentation for all
│   ├── soft_target.py           # Variant A
│   ├── kd_loss.py               # Variant B
│   └── train_kd.py              # PEdger training loop with minimal hooks for A/B
├── eval/
│   ├── run_bsds_eval.py         # NMS + ODS/OIS/AP
│   └── agreement_bins.py        # per-agreement-bin precision/recall (H2)
├── bench/speed_flops.py         # params, FLOPs, FPS, training GPU-hours
└── notebooks/kaggle_*.ipynb     # one notebook per stage
```

## Tasks

Do them in order. Each has a "done when" check; don't start the next stage until it holds.

### Task 1 — Evaluation first (`eval/run_bsds_eval.py`)

- NMS, then BSDS matching; ODS / OIS / AP; single- and multi-scale.
- `maxDist` 0.0075 (BSDS500, MultiCue), 0.011 (NYUDv2).
- MATLAB edges-toolbox as reference; Python port (e.g. pyEdgeEval) only once validated.
- **Done when:** a released PiDiNet or PEdger checkpoint reproduces its published ODS within ±0.002.

### Task 2 — Data preparation (`data/prepare_*.py`)

- BSDS500 + PASCAL VOC: identical lists and augmentation to PEdger (rotations, flips, rescaling).
- NYUDv2: train on train+val (795), test on 654.
- MultiCue: 80/20 split × 3 random splits with fixed seeds; separate edge and boundary labels.
- **Done when:** loaders return correct image/label pairs (visual check).

### Task 3 — Agreement maps (`data/agreement.py`)

```python
def human_trust(a, a_min=0.1):               # a = fraction of annotators marking edge
    agree = (2 * a - 1).abs()                # 1 = everyone agrees, 0 = evenly split
    return a_min + (1 - a_min) * agree       # α(p)
```

- Read `a(p)` from averaged GT PNGs (verify they are averaged, see [decisions.md Q2](decisions.md#open-questions)).
- Optional 1–2 px dilation of each annotator's map before averaging.
- **Done when:** maps look right on 10 sample images, with and without dilation.

### Task 4 — Teacher caching + teacher gate (`teachers/`)

- First check MEMO weights; if unavailable, implement `cache_diffusionedge.py` instead.
- Each `cache_<teacher>.py`: load the teacher from its official repo, run on **original** training images (and validation images for the gate), save probability maps as 16-bit PNG at the image's resolution. MuGE: save the mean over granularity outputs.

```python
def teacher_ensemble(maps, weighted=False):  # maps: list of (B,1,H,W) teacher probabilities
    T = torch.stack(maps)
    if not weighted:
        return T.mean(0)
    c = (2 * T - 1).abs() + 1e-6             # per-pixel confidence of each teacher
    return (c * T).sum(0) / c.sum(0)
```

- `teacher_gate.py`: evaluate each teacher and the ensemble on each dataset's validation split with Task 1's evaluator; write a table; output the list of teachers kept per dataset.
- **Done when:** teacher ODS table per dataset exists; teachers not better than PEdger on a dataset are dropped for it.

### Task 5 — PEdger baseline reproduction

- Run official PEdger training through our pipeline on BSDS + VOC (30 epochs, AdamW, 4-epoch warm-up to LR 0.001, linear decay, batch 16 via gradient accumulation).
- Record wall-clock and GPU-hours (sets the GPU budget, see [project-planning.md](project-planning.md#gpu-budget)).
- **Done when:** within ~0.005 of .821 ODS; otherwise our retrained number becomes the baseline.

### Task 6 — Joint augmentation (`pedger_kd/dataset.py`)

- Wrap PEdger's dataset; additionally load `T` (16-bit PNG → float in [0, 1]) and `a`.
- Sample **one** random crop / flip / rotation / scale and apply it to image, `Y`, `T`, `a`. Use bilinear interpolation for `T` and `a`; match PEdger's choice for `Y`.
- Returns `(image, Y, T, a)`.
- **Done when:** overlay check shows `T` aligned with the image after augmentation.

### Task 7 — Variant B, loss-level (`pedger_kd/kd_loss.py`)

```python
def standardize(z, eps=1e-6):
    return (z - z.mean((2, 3), keepdim=True)) / (z.std((2, 3), keepdim=True) + eps)

def kd_loss(z_s, p_t, w, tau=2.0, use_ls=True):
    z_t = torch.logit(p_t.clamp(1e-4, 1 - 1e-4))
    if use_ls:
        z_s, z_t = standardize(z_s), standardize(z_t)
    loss = F.binary_cross_entropy_with_logits(z_s / tau, torch.sigmoid(z_t / tau), reduction="none")
    return (w * loss).sum() / w.sum().clamp_min(1.0)
```

- Weight `w(p) = 1 − α(p) + w_min` (agreement) or `w = 1` (uniform).
- Total: `L = L_PEdger + λ_kd · kd_loss(...)`.
- Compute the KD term in float32 under AMP.
- **Done when:** unit test shows `λ_kd = 0` gives training identical to the baseline.

### Task 8 — Variant A, target-level (`pedger_kd/soft_target.py`)

```python
# Replaces PEdger's  Y_soft = eta*M + (1-eta)*Y
def soft_target(Y, M, T, alpha, gamma=0.5, ramp=1.0):
    model_view = (1 - gamma) * M + gamma * T          # self-ensemble + external teachers
    trust_models = (1 - alpha) * ramp                 # ramp = ep/EP keeps early epochs label-driven
    return (1 - trust_models) * Y + trust_models * model_view
```

- `gamma`: scalar or per-pixel tensor `|2T−1| / (|2T−1| + |2M−1| + eps)`.
- `ramp`: `ep / EP` or `1.0`.
- NYUDv2: `alpha` from constant schedule (`1 − η_ep`) or teacher confidence (`1 − c·|2T−1|`).
- Hook into the point in `train_kd.py` where PEdger forms its soft target.
- **Done when:** unit test shows `gamma = 0`, `ramp = 1`, `alpha = 1 − eta` reproduces PEdger's target exactly.

### Task 9 — Training runs (`pedger_kd/train_kd.py`, `configs/`)

- Runs: baseline; B-uniform; B-agree; A-agree with each single teacher and the ensemble; γ sweep; 3 seeds for main results.
- AMP; checkpoint every epoch; resume from latest checkpoint; log config, seed, git commit and GPU-hours.
- **Done when:** all runs logged with GPU-hours.

### Task 10 — Evaluation and analysis (`eval/`)

- ODS / OIS / AP for SS and MS, with and without VOC; NYUDv2; MultiCue mean ± std over 3 splits.
- `agreement_bins.py`: split BSDS500 test edge pixels into low / medium / high agreement; precision and recall per bin for PEdger, PEdger++ and ours.
- Optional crispness-aware metrics (CEval / AC).
- **Done when:** results tables filled.

### Task 11 — Efficiency and cost (`bench/speed_flops.py`)

- Params; FLOPs via `fvcore` or `thop`; FPS at batch 1 after warm-up, fixed input (e.g. 320×480), on T4 and CPU.
- Training GPU-hours from logs for PEdger, ours and PEdger++.
- **Done when:** efficiency and training-cost table exists; ours equals PEdger on params / FLOPs / FPS.

### Task 12 — Optional: PEdger++ + ours

- Plug `soft_target` into efficient PEdger++ training (same `Ỹ = η·M + (1−η)·Y` hook).
- **Done when:** one result row for H5.

## Config schema

One YAML per experiment in `configs/`. Suggested keys:

```yaml
name: A_agree_ensemble_bsds_voc
dataset: bsds            # bsds | nyud | multicue
use_voc: true
multicue_split: null     # 0 | 1 | 2 for MultiCue
seed: 0

variant: A               # baseline | A | B
teachers: [sauge, muge, memo]
ensemble: mean           # mean | confidence

# agreement
alpha_mode: agreement    # agreement | constant | teacher_conf  (NYUD uses the latter two)
alpha_min: 0.1
dilation_px: 0           # 0, 1 or 2
conf_c: 1.0              # c in alpha = 1 - c*|2T-1|

# Variant A
gamma: 0.5               # float or "confidence"
ramp: true               # multiply (1 - alpha) by ep/EP

# Variant B
lambda_kd: 1.0
tau: 2.0
use_ls: true
w_mode: agreement        # agreement | uniform
w_min: 0.1

# training (PEdger defaults)
epochs: 30
lr: 0.001
warmup_epochs: 4
batch_size: 16           # via gradient accumulation
amp: true
```

## Kaggle notebooks

| Notebook | Stage | Tasks |
|---|---|---|
| `kaggle_01_eval_check.ipynb` | Reproduce published ODS | 1 |
| `kaggle_02_data_agreement.ipynb` | Data lists, agreement maps | 2, 3 |
| `kaggle_03_teachers.ipynb` | Cache teachers, run gate | 4 |
| `kaggle_04_baseline.ipynb` | PEdger baseline | 5 |
| `kaggle_05_train_kd.ipynb` | Variant A / B runs (config-driven) | 6–9 |
| `kaggle_06_eval_bench.ipynb` | Evaluation, agreement bins, efficiency | 10, 11 |
| `kaggle_07_pedgerpp.ipynb` | Optional PEdger++ + ours | 12 |

Datasets and teacher caches are attached as Kaggle Datasets; checkpoints and logs are written to notebook output.

## Status

- [ ] 1 Evaluation reproduced
- [ ] 2 Data preparation
- [ ] 3 Agreement maps
- [ ] 4 Teacher caching + gate
- [ ] 5 PEdger baseline
- [ ] 6 Joint augmentation
- [ ] 7 Variant B
- [ ] 8 Variant A
- [ ] 9 Training runs
- [ ] 10 Evaluation and analysis
- [ ] 11 Efficiency and cost
- [ ] 12 Optional PEdger++ + ours
