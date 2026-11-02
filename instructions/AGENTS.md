# AGENTS.md

Instructions for AI coding agents (and human contributors) working in this repository.

## Project in three lines

- We distil three frozen foundation-model edge detectors (SAUGE, MuGE, MEMO; DiffusionEdge as backup) into **PEdger**, a 0.73M-parameter edge detector.
- Per-pixel **human annotator agreement** decides how much the training target follows the human label vs the models.
- The student's architecture, parameter count and inference cost must not change.

Source of truth: [Edge_Detection_Research_Plan.md](../Edge_Detection_Research_Plan.md). If anything here conflicts with it, the plan wins; flag the conflict.

## Hard rules

1. **Never edit `third_party/PEdger/` or `third_party/PEdgerPlus/`.** Wrap, subclass or hook from `pedger_kd/`. Our baseline must be the official code.
2. **The student stays identical at inference.** No new layers, heads or parameters in the deployed network. Params / FLOPs / FPS must equal PEdger's (checked by `bench/speed_flops.py`).
3. **Teachers run once, offline.** Teacher models are never loaded in the training loop. Their probability maps are cached as **16-bit PNG** on the *original* (un-augmented) training images.
4. **Same geometric augmentation for everything.** Image, GT `Y`, teacher map `T` and agreement map `a(p)` must receive the identical random crop / flip / rotate / scale.
5. **No test data in teacher work.** Teacher caching uses training images; the teacher gate uses validation splits only.
6. **Evaluation must be validated.** Use the MATLAB BSDS / edges-toolbox benchmark, or a Python port only after it reproduces a published number within ±0.002. `maxDist` = 0.0075 (BSDS500, MultiCue), 0.011 (NYUDv2).
7. **Never mix settings in one column:** single-scale vs multi-scale, with vs without PASCAL VOC.
8. **Kaggle-safe training:** mixed precision (`torch.cuda.amp`); every long job must survive a session or quota cut-off. Use `pedger_kd/checkpoint.py` (full state, every epoch + every ~20 min, atomic writes, time-budget stop, auto-resume). Never write a training loop that cannot resume. Log GPU-hours accumulated across sessions. Design: [architecture.md §7](architecture.md#7-checkpointing-and-resume).
9. **Seeds:** 3 seeds for main results, 1 seed for ablations/sweeps. Seeds go in the config.

## Required equivalence tests

These must pass before any training result is trusted:

- **Variant B:** with `lambda_kd = 0`, the loss and gradients equal the unmodified PEdger loss.
- **Variant A:** with `gamma = 0` and `alpha = 1 − eta_ep`, `soft_target(...)` reproduces PEdger's `Ỹ = η·M + (1−η)·Y` exactly.
- **Augmentation:** overlay test showing `T` and `a(p)` stay aligned with the image after augmentation.

## Where things go

| Path | Contents |
|---|---|
| `configs/` | One YAML per experiment (baseline, B_uniform, B_agree, A_*, single / ensemble teachers) |
| `data/` | `prepare_bsds.py`, `prepare_nyud.py`, `prepare_multicue.py`, `agreement.py` |
| `teachers/` | `cache_<teacher>.py`, `ensemble.py`, `teacher_gate.py` |
| `pedger_kd/` | `dataset.py`, `soft_target.py`, `kd_loss.py`, `train_kd.py` |
| `eval/` | `run_bsds_eval.py`, `agreement_bins.py` |
| `bench/` | `speed_flops.py` |
| `notebooks/` | `kaggle_*.ipynb`, one per stage |

Details per module: [implementation.md](implementation.md). Design: [architecture.md](architecture.md).

## Conventions

- Python + PyTorch. Tensors are `(B, 1, H, W)` probability maps in `[0, 1]` unless named `z_*` (logits).
- Every hyperparameter lives in the YAML config, not hard-coded. Config keys are listed in [implementation.md](implementation.md#config-schema).
- Names follow the plan's notation: `Y`, `M`, `T`, `a`, `alpha`, `gamma`, `eta`, `ramp`, `tau`, `lambda_kd`, `w_min`, `alpha_min`.
- Clamp before `logit` (`1e-4`), add `eps` before any division.
- Keep changes small and tied to a task number from [implementation.md](implementation.md).

## Keep the docs current

- A design choice that differs from the plan → add an entry to [decisions.md](decisions.md).
- A task finished → tick it in [implementation.md](implementation.md#status).
- A schedule slip or new risk → update [project-planning.md](project-planning.md).
- Before opening a PR → go through [code-review.md](code-review.md).

## Verify before claiming

These are unconfirmed in the plan. Check and record the answer in [decisions.md](decisions.md#open-questions) before relying on them:

- Whether **MEMO** weights are released (else switch to DiffusionEdge).
- That the downloaded **BSDS500 augmented GT PNGs are averaged** (fraction of annotators), not binary.
- The exact **number of annotations per MultiCue image**.
- Any number copied into a results table: re-check it against the original paper.
