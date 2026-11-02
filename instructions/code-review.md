# Code Review

Checklist for reviewing changes in this repository. General rules are in [AGENTS.md](AGENTS.md); the design they protect is in [architecture.md](architecture.md).

Mark each item ✅ (checked), ❌ (problem) or N/A in the review.

## 1. Scope

- [ ] The change is tied to a task in [implementation.md](implementation.md) and does one thing.
- [ ] Nothing under `third_party/` was modified (`git diff --stat third_party/` is empty).
- [ ] Any departure from the research plan has a matching entry in [decisions.md](decisions.md).

## 2. Method correctness

- [ ] **Variant A equivalence test passes:** `gamma = 0`, `ramp = 1`, `alpha = 1 − eta` reproduces PEdger's `Ỹ = η·M + (1−η)·Y` exactly.
- [ ] **Variant B equivalence test passes:** `lambda_kd = 0` gives the unmodified PEdger loss and gradients.
- [ ] `α(p) = α_min + (1 − α_min)·|2a − 1|` is implemented as written; `a` is the averaged GT, not a binarised one.
- [ ] The direction of trust is right: high agreement → follow `Y`; low agreement → follow `M`/`T`.
- [ ] `γ` mixes `M` and `T` the right way round (`γ = 1` → teachers only).
- [ ] The `ep/EP` ramp uses the current epoch, not the step, and can be disabled by config.
- [ ] Ensemble only uses teachers that passed the gate for this dataset.

## 3. Student stays unchanged

- [ ] No teacher model, `T` or `a(p)` is referenced in the inference path.
- [ ] No new modules or parameters added to the deployed network.
- [ ] `bench/speed_flops.py` shows params / FLOPs equal to PEdger's.

## 4. Data and augmentation

- [ ] Image, `Y`, `T` and `a` receive the **same** random geometric transform (one sampled transform, applied to all four).
- [ ] Interpolation is appropriate (bilinear for `T` and `a`; matches PEdger for `Y`).
- [ ] 16-bit teacher PNGs are decoded to float in [0, 1] (divide by 65535, not 255).
- [ ] Train / val / test lists match PEdger's; MultiCue splits use fixed seeds.
- [ ] **No leakage:** teacher caching uses training images; the teacher gate uses validation splits only; nothing tuned on test.

## 5. Numerics

- [ ] Probabilities clamped (e.g. `1e-4`) before `torch.logit`.
- [ ] `eps` before every division (`standardize`, confidence weights, `γ(p)`, weighted loss normaliser).
- [ ] KD loss and soft-target arithmetic done in float32 under AMP.
- [ ] Logit standardization is per image over spatial dims `(2, 3)`, not over the batch.
- [ ] No NaN/Inf in a short smoke run (log loss and target stats for one epoch).

## 6. Evaluation

- [ ] Evaluator validated against a published number (±0.002) before use.
- [ ] `maxDist` is 0.0075 for BSDS500 / MultiCue and 0.011 for NYUDv2.
- [ ] NMS applied before matching (except crispness-aware metrics).
- [ ] SS and MS, and ± VOC, reported in separate columns.
- [ ] MultiCue reported as mean ± std over the 3 splits.
- [ ] FPS measured at batch 1, after warm-up, at a fixed stated input size, on T4 and CPU.

## 7. Reproducibility and Kaggle robustness

- [ ] All hyperparameters come from the YAML config; no magic numbers in code.
- [ ] Seed set for Python, NumPy and PyTorch; seed stored in the run log.
- [ ] Run log records config, git commit, seed and GPU-hours.
- [ ] Checkpoint saved every epoch; resuming from a checkpoint restores model, momentum copies, optimizer, scheduler, AMP scaler and epoch.
- [ ] Paths to datasets and caches are configurable (Kaggle input vs local).

## 8. Tests

- [ ] Unit tests for `human_trust`, `teacher_ensemble`, `soft_target`, `kd_loss` (including the equivalence cases).
- [ ] Augmentation alignment test (overlay or synthetic-pattern check).
- [ ] Tests run on CPU without the datasets (small synthetic tensors).

## 9. Results hygiene

- [ ] Any number added to a results table is traceable to a run log or to a cited paper.
- [ ] Published numbers copied from papers were re-checked against the original paper.
- [ ] Claims in text match the evidence (e.g. "gain beyond seed variation" backed by 3 seeds).

## Review outcome

| Field | Value |
|---|---|
| Task(s) | |
| Reviewer | |
| Equivalence tests | pass / fail |
| Decision | approve / request changes |
| Notes | |
