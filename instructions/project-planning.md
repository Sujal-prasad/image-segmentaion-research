# Project Planning

Timeline, milestones, compute budget and risks. Task numbers refer to [implementation.md](implementation.md); hypotheses H1–H6 are defined in [requirements.md](requirements.md#7-acceptance-criteria-hypotheses).

## Goals

1. Show that external foundation-model teachers improve PEdger at zero inference cost (H1).
2. Show that agreement-aware weighting beats uniform distillation, with gains in low-agreement pixels (H2).
3. Show that a multi-teacher ensemble beats any single teacher (H3).
4. Compare external teaching against PEdger++'s richer self-teaching, including training cost (H4).
5. *(Optional)* Show the two add up (H5).

## Five-week timeline

| Week | Work | Tasks | Deliverable |
|---|---|---|---|
| 1 | Reproduce evaluation; prepare data; build agreement maps; check MEMO weights; cache all teachers; run teacher gate. | 1, 2, 3, 4 | Validated eval; loaders; agreement maps; teacher ODS table per dataset |
| 2 | Reproduce PEdger baseline; joint augmentation; implement and unit-test Variants B and A. | 5, 6, 7, 8 | Baseline ODS; passing equivalence tests |
| 3 | BSDS500: single-teacher and ensemble runs, Variants A and B. | 9 | First BSDS500 results table |
| 4 | Ablations; final 3-seed runs; NYUDv2 and MultiCue (3 splits). | 9, 10 | Main results tables for all datasets |
| 5 | Agreement-bin analysis; efficiency and training cost; optional PEdger++ + ours; figures and writing. | 10, 11, 12 | Paper draft |

## Milestones and gates

| Milestone | Gate (must hold before moving on) | Target |
|---|---|---|
| M1 Eval reproduced | Published ODS reproduced within ±0.002 | End of week 1 |
| M2 Teachers ready | All caches done; teacher gate table filled; Q1–Q3 in [decisions.md](decisions.md#open-questions) answered | End of week 1 |
| M3 Baseline reproduced | PEdger within ~0.005 of .821, or retrained number adopted as baseline | Week 2 |
| M4 Methods verified | Equivalence tests for Variants A and B pass; augmentation overlay check passes | End of week 2 |
| M5 BSDS500 results | Single-teacher, ensemble, A vs B runs complete | End of week 3 |
| M6 All main results | 3 seeds for main rows; NYUDv2 and MultiCue done | End of week 4 |
| M7 Paper draft | Agreement bins, efficiency table, figures, text | End of week 5 |

## GPU budget

- Budget: ~30 GPU-h per week on Kaggle T4.
- **Time the first full PEdger run first.** If one run takes *T* hours, the week allows about **30 / T** runs.
- Priority order:
  1. Main results with 3 seeds (baseline, B-uniform, A-ensemble).
  2. Single-teacher runs (1 seed, then 3 for the best).
  3. Ablations (1 seed each).
  4. NYUDv2 / MultiCue.
  5. Optional PEdger++ + ours.
- Log GPU-hours for every run; they are reported in the paper.

## Experiment matrix

### Main results

| Setting | BSDS500 (BSDS / +VOC, SS / MS) | NYUDv2 | MultiCue edge / boundary |
|---|---|---|---|
| PEdger (published) | ✓ | ✓ | ✓ (from PEdger++ paper) |
| PEdger (our retrain) | ✓ | ✓ | ✓ |
| + Variant B, uniform, SAUGE | ✓ | ✓ | ✓ |
| + Variant A, agreement-aware, each single teacher | ✓ | teacher-confidence | ✓ |
| **+ Variant A, agreement-aware, teacher ensemble (ours, full)** | ✓ | teacher-confidence | ✓ |
| PEdger++ (published) | ✓ | ✓ | ✓ |
| PEdger++ + ours (optional) | ✓ | – | – |
| Teachers: SAUGE, MuGE, MEMO, ensemble | ✓ | ✓ (new zero-shot numbers) | ✓ (new boundary numbers) |

### Ablations (BSDS500, 1 seed)

- Teacher: SAUGE / MuGE / MEMO / mean ensemble / confidence-weighted ensemble.
- `α_min`; agreement-map dilation on/off.
- `γ` ∈ {0, 0.5, 1, confidence-based}; `ep/EP` ramp on/off.
- Variant B: `λ_kd`, `τ`, logit standardization on/off.

## Risk register

| Risk | Trigger | Fallback |
|---|---|---|
| MEMO weights not released | Q1 answered "no" | Use DiffusionEdge as the third teacher. |
| Teacher weak zero-shot on NYUDv2 / MultiCue | Fails teacher gate | Drop it for that dataset; on NYUD rely on DiffusionEdge's NYUD model and/or fine-tune SAUGE's 1.3M side network (SAM frozen). |
| DiffusionEdge maps too binary | Little gain from it as teacher | Light Gaussian blur or lower ensemble weight; report the effect. |
| PEdger retrain below .821 | M3 gate missed | Report our retrained number as the baseline; state it openly. |
| PEdger + ours does not beat PEdger++ | M6 results | Lead with H1–H3, agreement bins and training-cost comparison; add PEdger++ + ours (H5). |
| Python eval differs from MATLAB | M1 gate missed | Use MATLAB; validate any port on released checkpoints. |
| Kaggle quota / session limits | Runs cut off | Per-epoch checkpoints; caches as Kaggle Datasets; 1 seed for ablations. |
| Someone publishes the same idea | Literature check | Re-check Google Scholar and arXiv right before submission; sharpen the novelty claim. |

## Pre-submission checklist

- [ ] Every number re-checked against its original paper.
- [ ] Annotation format of each downloaded dataset confirmed and recorded.
- [ ] Literature search repeated for new work on distillation for edge detection.
- [ ] SS / MS and ± VOC never mixed in one column.
- [ ] Training GPU-hours reported for PEdger, ours and PEdger++.
- [ ] Params / FLOPs / FPS of ours shown equal to PEdger.
- [ ] Novelty claim stated precisely: agreement used to decide **where the student trusts its teachers**.
