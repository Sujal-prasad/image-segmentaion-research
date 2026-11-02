# Agreement-Aware Multi-Teacher Distillation for Lightweight Edge Detection

> We teach **PEdger**, a 0.73M-parameter edge detector, using three recent state-of-the-art edge detectors (**SAUGE**, **MuGE** and **MEMO**), and let human annotator disagreement decide, pixel by pixel, when PEdger should trust the teachers instead of the human labels.

**Status:** planning. No code has been written yet. The full research plan is in [Edge_Detection_Research_Plan.md](../Edge_Detection_Research_Plan.md), which is the source of truth for every number and decision in these docs.

---

## The problem in brief

Edge detectors face a size/accuracy trade-off:

| | Large models | Compact models |
|---|---|---|
| Examples | SAUGE, MEMO, MuGE, EasyControlEdge, EDTER | PEdger, PEdger++, PiDiNet |
| Size | 40M – 470M+ params, or a frozen foundation model | under 1M params |
| BSDS500 ODS | about .83 – .86 | about .81 – .835 |
| Usable on phones / robots | No | Yes |

PEdger and PEdger++ learn only from their own averaged predictions. Neither has ever been taught by an external, stronger model, and both throw away the information about where human annotators disagree.

## Research questions

1. Can distillation from recent foundation-model edge detectors improve PEdger with **no extra parameters and no extra inference cost**?
2. Does using **human annotator agreement** to steer that distillation help more than distillation alone?
3. Is **external teaching** more effective than richer **self-teaching** (PEdger + ours vs PEdger++)?

## Setup at a glance

| Role | Models |
|---|---|
| **Student** (the only model we train) | PEdger (ACM MM 2023, 0.73M params) |
| **Teachers** (frozen, predictions cached once) | SAUGE (AAAI 2025), MuGE (CVPR 2024), MEMO (CVPR 2026); backup: DiffusionEdge (AAAI 2024) |
| **Main comparison** | PEdger (published + our retrain), PEdger++ |
| **Optional final experiment** | PEdger++ + our method |
| **Datasets** | BSDS500 (+ PASCAL VOC aug), NYUDv2, MultiCue |
| **Compute** | Kaggle T4 (16 GB), ~12 h sessions, ~30 GPU-h/week |

### The core idea

```
PEdger original:  Ỹ(p) = η_ep · M(p) + (1 − η_ep) · Y(p)
Ours (Variant A): Ỹ(p) = α(p) · Y(p) + (1 − α(p)) · [ (1 − γ) · M(p) + γ · T(p) ]
```

`Y` = human label, `M` = PEdger's self-ensemble, `T` = teacher ensemble, `α(p)` = per-pixel human-trust weight from annotator agreement. Where humans agree, follow the label; where they disagree, follow the models. Inference is the unchanged PEdger network.

## Targets (hypotheses, not promises)

| Dataset | PEdger (published) | Realistic aim | Ambitious headline |
|---|---|---|---|
| BSDS500 (w/ VOC) | .821 / .840 | .828 – .834 | ≥ .835 (matching or beating PEdger++) |
| NYUDv2 | .742 / .757 | .750 – .760 | ≥ .765 (matching PEdger++) |
| MultiCue boundary | .828 / .839 | .840 – .855 | ≥ .866 (matching PEdger++) |

## Planned repository layout

```
pedger-mtkd/
├── third_party/        # official PEdger (+ PEdger++) code, unmodified submodules
├── configs/            # one YAML per experiment
├── data/               # dataset prep + agreement maps
├── teachers/           # one-off teacher caching, ensembling, teacher gate
├── pedger_kd/          # dataset wrapper, Variant A soft target, Variant B KD loss, training loop
├── eval/               # BSDS-style ODS/OIS/AP + agreement-bin analysis
├── bench/              # params, FLOPs, FPS, GPU-hours
└── notebooks/          # one Kaggle notebook per stage
```

See [architecture.md](architecture.md) for how the pieces fit together.

## Quick start (once code exists)

1. Clone with submodules: `git clone --recursive <repo>` (pulls official PEdger into `third_party/`).
2. Upload BSDS500, NYUDv2, MultiCue and the cached teacher maps as **Kaggle Datasets**.
3. Run the notebooks in order: evaluation check → data prep → teacher caching + gate → baseline → KD runs → evaluation.
4. Each experiment is driven by a YAML in `configs/`; see [implementation.md](implementation.md) for the config schema.

## Documentation

| Doc | What it covers |
|---|---|
| [Edge_Detection_Research_Plan.md](../Edge_Detection_Research_Plan.md) | Full research plan: scoreboard, related work, method, references |
| [AGENTS.md](AGENTS.md) | Rules for AI coding agents and contributors |
| [architecture.md](architecture.md) | System design, data flow, formulas |
| [requirements.md](requirements.md) | Functional / non-functional requirements, dependencies, acceptance criteria |
| [decisions.md](decisions.md) | Decision log (why PEdger, why these teachers, etc.) |
| [project-planning.md](project-planning.md) | Five-week timeline, milestones, GPU budget, risks |
| [implementation.md](implementation.md) | Module-by-module tasks, function signatures, config schema, status |
| [code-review.md](code-review.md) | Review checklist specific to this project |

## License

See [LICENSE](../LICENSE). Third-party code under `third_party/` keeps its original license.
