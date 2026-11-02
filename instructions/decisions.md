# Decisions

A log of the project's key design decisions, why they were made, and what they cost. Add a new entry whenever a choice departs from or extends [Edge_Detection_Research_Plan.md](../Edge_Detection_Research_Plan.md). Statuses: **Accepted**, **Proposed**, **Superseded**, **Pending verification**.

---

### D1. Student: PEdger, not PEdger++

- **Status:** Accepted
- **Context:** We need a sub-1M-parameter student with public code.
- **Decision:** Train PEdger (ACM MM 2023, 0.73M params) in all main experiments.
- **Why:** 2–5 points of headroom to the teachers (PEdger++ is within ~0.7 of SAUGE on BSDS500); peer reviewed (PEdger++ is a preprint); cheaper on a T4 (no parameter sampling or validation-fitted ensemble); and PEdger → PEdger++ (more self-teaching) vs PEdger → ours (external teaching) answers RQ3 directly.
- **Alternatives:** PEdger++ (less headroom), PiDiNet (weaker, no self-teaching hook), TEED.
- **Consequences:** PEdger + ours may still fall short of PEdger++; see D13 and the risk register in [project-planning.md](project-planning.md#risk-register).

### D2. Teachers: SAUGE, MuGE, MEMO; DiffusionEdge as backup

- **Status:** Accepted (MEMO pending verification)
- **Decision:** Use SAUGE (SAM-based, AAAI 2025), MuGE (CNN, CVPR 2024), MEMO (DINOv2-based, CVPR 2026). DiffusionEdge (AAAI 2024) replaces MEMO if its weights are unavailable.
- **Why:** Recent (2024–2026); diverse foundations so their errors differ; all clearly stronger than PEdger on BSDS500; weights released or checkable. DiffusionEdge is the only candidate with released NYUD weights.
- **Consequences:** DiffusionEdge outputs are nearly binary (NYUD AP .508), so it may help less as a soft teacher (fallback: light Gaussian blur or lower weight).

### D3. Excluded teacher candidates

- **Status:** Accepted
- **Decision:** Exclude GED and EasyControlEdge (no code/weights), RankED (no code repo), EDMB (code availability not confirmed), UAED (earlier version of MuGE, adds little diversity).
- **Consequences:** They appear only as published comparison rows.

### D4. Ensemble: mean by default, confidence-weighted as ablation

- **Status:** Accepted
- **Decision:** `T = mean(T_SAUGE, T_MuGE, T_MEMO)`; ablate per-pixel weighting by `|2T_k − 1|`. MuGE contributes the average of its granularity outputs.
- **Why:** Simplest defensible default; ablation tests whether confidence weighting helps.

### D5. Variant A (target-level) is the main method; Variant B (loss-level) is the baseline

- **Status:** Accepted
- **Decision:** Main method replaces PEdger's soft target with the agreement-aware target. Variant B adds a separate weighted KD loss and leaves PEdger untouched.
- **Why:** Variant A integrates naturally with PEdger's existing soft-target mechanism; Variant B is the standard approach. Variant A must beat B to justify itself.
- **Consequences:** Both are implemented and reported.

### D6. Agreement from averaged BSDS GT

- **Status:** Accepted (GT format pending verification)
- **Decision:** `a(p)` = fraction of annotators, read from the augmented BSDS GT PNGs. `agree = |2a − 1|`, `α = α_min + (1 − α_min)·agree`. Optional 1–2 px dilation, ablated.
- **Why:** Agreement information is already in the data (PEdger++ binarises it at 0.2, which implies averaged maps); no extra annotation needed.

### D7. Teachers run once offline, cached as 16-bit PNG

- **Status:** Accepted
- **Decision:** Run each teacher once on original (un-augmented) training images; store probability maps as 16-bit PNG; apply augmentation jointly at load time.
- **Why:** Teachers are large (SAM, DINOv2, diffusion); running them every step would not fit the T4 budget.
- **Consequences:** Requires joint augmentation of image, `Y`, `T`, `a` (implementation task 6).

### D8. Per-dataset teacher gate

- **Status:** Accepted
- **Decision:** Before use, score each teacher and the ensemble on each dataset's validation split. A teacher not better than PEdger on a dataset is dropped for that dataset.
- **Why:** SAUGE, MuGE and MEMO have BSDS weights only; on NYUDv2 and MultiCue they run zero-shot.

### D9. Build on the official PEdger code, unmodified

- **Status:** Accepted
- **Decision:** PEdger (and PEdger++) as unmodified submodules under `third_party/`; our code hooks in from `pedger_kd/`.
- **Why:** Guarantees the baseline is the authors' method; keeps diffs reviewable.

### D10. MATLAB benchmark is the evaluation reference

- **Status:** Accepted
- **Decision:** Use the standard MATLAB BSDS / edges-toolbox benchmark. A Python port is allowed only after reproducing a published number within ±0.002.
- **Why:** Small evaluation differences can be as large as the gains we are measuring.

### D11. Compute: Kaggle T4 with AMP

- **Status:** Accepted
- **Decision:** All runs on Kaggle (T4 16 GB, ~12 h sessions, ~30 GPU-h/week) with mixed precision and per-epoch checkpoints. Data and caches stored as Kaggle Datasets.
- **Consequences:** Limits the number of runs; 3 seeds only for main results, 1 seed for sweeps.

### D12. NYUDv2: no agreement map, two α options

- **Status:** Accepted
- **Decision:** Test (a) constant α following PEdger's η schedule and (b) teacher-confidence α, `α(p) = 1 − c·|2T(p) − 1|`. Include DiffusionEdge's NYUD model in the ensemble if it passes the gate.

### D13. PEdger++ + ours is optional

- **Status:** Accepted
- **Decision:** Run only if time allows, using efficient PEdger++ training.
- **Why:** Shows complementarity (H5), and is a fallback contribution if PEdger + ours does not beat PEdger++.

### D14. Reporting rules

- **Status:** Accepted
- **Decision:** Report SS / MS and ± VOC separately; MultiCue as mean ± std over 3 splits; training GPU-hours for every compared method; plain-distillation baseline with equal GPU time.
- **Why:** Fair-comparison rule from the canonical KD study [34]; needed for the PEdger++ comparison.

---

## Open questions

Resolve these early (week 1) and record the answer here.

| # | Question | Affects | Answer |
|---|---|---|---|
| Q1 | Are MEMO's weights released? | D2 (teacher 3) | _pending_ |
| Q2 | Are the downloaded BSDS augmented GT PNGs averaged (not binary)? | D6 (agreement maps) | _pending_ |
| Q3 | How many annotations per MultiCue image? | Agreement on MultiCue | _pending_ |
| Q4 | Does PEdger's repo release checkpoints? | Task 1 / 5 | _pending_ |
| Q5 | Time for one full PEdger run on a T4 (T hours)? | GPU budget (≈ 30/T runs per week) | _pending_ |
