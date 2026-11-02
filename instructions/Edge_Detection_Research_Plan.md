# Agreement-Aware Multi-Teacher Distillation from Foundation-Model Edge Detectors for Lightweight Edge Detection

**Research plan** · Object boundary detection in natural scenes · November 2026

> **In one sentence:** We teach **PEdger**, a 0.73M-parameter edge detector, using three recent state-of-the-art edge detectors (SAUGE, MuGE and MEMO), and let human disagreement decide, pixel by pixel, when PEdger should trust the teachers instead of the human labels.

**Scope decisions:**

- **Student:** PEdger (ACM MM 2023, 0.73M parameters). It is the only model we train in the main experiments.
- **Teachers:** SAUGE (AAAI 2025), MuGE (CVPR 2024) and MEMO (CVPR 2026), with DiffusionEdge (AAAI 2024) as a backup.
- **Comparison:** PEdger's published results, **PEdger++** (the authors' newer self-teaching version), and the state-of-the-art models below.
- **Optional final experiment:** apply our method to PEdger++ as well, to show the two ideas add up.

---

## Contents

1. [The Problem](#1-the-problem)
2. [Datasets](#2-datasets)
3. [Benchmarks and Evaluation Protocol](#3-benchmarks-and-evaluation-protocol)
4. [Models](#4-models)
5. [Current Results (the Scoreboard)](#5-current-results-the-scoreboard)
6. [Previous Work in Our Research Area](#6-previous-work-in-our-research-area)
7. [The Research Gap](#7-the-research-gap)
8. [Our Method](#8-our-method)
9. [Hypotheses and Expected Results](#9-hypotheses-and-expected-results)
10. [What We Have to Code](#10-what-we-have-to-code)
11. [Experiments and Timeline](#11-experiments-and-timeline)
12. [Risks and Fallbacks](#12-risks-and-fallbacks)
13. [References](#13-references)

---

## 1. The Problem

Edge detection means tracing the meaningful outlines in an image. An **edge** is any sudden change in brightness, colour or texture; a **boundary** is the outline of an actual object. Natural scenes are hard because shadows, textures (grass, brick, fur), occlusion and low contrast create misleading edges or hide real ones, and even humans disagree about where some boundaries lie.

The field faces a clear trade-off:

| | Large models | Compact models |
|---|---|---|
| Examples | SAUGE, MEMO, MuGE, EasyControlEdge, EDTER | PEdger, PEdger++, PiDiNet |
| Size | 40M – 470M+ parameters, or a frozen foundation model | under 1M parameters |
| BSDS500 ODS | about .83 – .86 | about .81 – .835 |
| Usable on phones / robots | No | Yes |

PEdger trains itself on soft labels built from its own averaged predictions. PEdger++ extends this with more self-generated predictions. Neither has ever been taught by an external, stronger model.

**Research questions:**

1. Can distillation from recent foundation-model edge detectors improve PEdger with no extra parameters and no extra inference cost?
2. Does using human annotator agreement to steer that distillation help more than distillation alone?
3. Is learning from external teachers more effective than richer self-teaching, i.e. how does PEdger + our method compare with PEdger++?

---

## 2. Datasets

| Dataset | Content | Size and standard split | Annotations | Role in our study |
|---|---|---|---|---|
| **BSDS500** [1] | Natural outdoor and everyday photos | 500 images: 200 train, 100 val, 200 test. Train on train+val (augmented), test on 200. | **Several annotators per image** (typically about 5) | Main benchmark, and the source of our agreement maps |
| **PASCAL VOC Context (augmentation)** | Extra natural images | About 10k images, training only | Single annotation | Extra training data; PEdger's headline .821 uses it |
| **NYUDv2** [2] | Indoor RGB-D rooms | 1,449 images: 381 train, 414 val, 654 test. Train on train+val (795). | Single annotation | Second benchmark |
| **MultiCue** [3] | Short stereo video scenes | 100 scenes: 80 train / 20 test, **3 random splits**, averaged | Separate **edge** and **boundary** labels, several annotations per image (verify the exact count when downloading) | Third benchmark |

**Augmentation:** follow PEdger's own data pipeline exactly (the standard augmented BSDS, PASCAL and NYUD sets: rotations, flips, rescaling) so our baseline matches theirs.

**Agreement maps come for free on BSDS500.** The augmented BSDS ground-truth PNGs store the fraction of annotators marking each pixel; PEdger++ binarises them at 0.2, which confirms they are averaged rather than binary. We use these averaged maps to compute agreement. **Verify this on the files you download.**

---

## 3. Benchmarks and Evaluation Protocol

### Metrics

| Metric | Meaning | Note |
|---|---|---|
| **ODS** (Optimal Dataset Scale) | F-score with one threshold for the whole dataset | **The headline number** |
| **OIS** (Optimal Image Scale) | F-score with the best threshold per image, averaged | Always ≥ ODS |
| **AP** (Average Precision) | Area under the precision–recall curve | PEdger does not report it; we report it for our runs |
| **Params / FLOPs / FPS** | Model size, computation, speed | Must stay identical to PEdger |
| **Training cost (GPU-hours)** | Total training time | Needed to compare fairly with PEdger++ |
| **CEval / AC** (crispness-aware evaluation, average crispness) | Scores without thinning post-processing | Optional; reported by DiffusionEdge, MEMO, EasyControlEdge |

### Protocol details

- **Post-processing:** standard non-maximum suppression (NMS), then matching against the ground truth.
- **Tolerance (`maxDist`):** **0.0075** for BSDS500 and MultiCue, **0.011** for NYUDv2.
- **Tool:** the standard MATLAB BSDS / edges-toolbox benchmark. A Python port (for example pyEdgeEval) is acceptable only after it reproduces a published number.
- **Settings:** report single-scale (SS) and multi-scale (MS), and with and without PASCAL VOC, separately; never mix them in one comparison column.
- **MultiCue:** mean ± standard deviation over 3 splits.
- **Speed:** batch size 1, after warm-up, fixed input size (for example 320×480 for BSDS), on the T4 and on a CPU. FLOPs via `fvcore` or `thop`.

---

## 4. Models

### 4.1 Student: PEdger (the model we train)

| Property | Value |
|---|---|
| Paper | Practical Edge Detection via Robust Collaborative Learning (PEdger) [22], ACM Multimedia 2023 (peer reviewed) |
| Size and speed | 0.73M parameters, trained from scratch (no ImageNet), ~92 FPS at 200×200 on an RTX 3090 |
| Published results | BSDS500 .821 / .840 (with VOC), NYUDv2 .742 / .757 |
| Code | [ForawardStar/PEdger](https://github.com/ForawardStar/PEdger) (check for released checkpoints; plan to retrain the baseline anyway) |

How PEdger trains (our method plugs into this):

1. **Two networks train together:** a recurrent network and a non-recurrent network.
2. **Momentum copies** of each are updated every epoch: `W_m ← (W_bp + W_m) / 2`.
3. Their predictions are **fused per pixel by confidence** (distance from 0.5) into **M**.
4. **Soft target:** `Ỹ = η_ep · M + (1 − η_ep) · Y`, where `η_ep = 0.8 · ep / EP` grows with the epoch. Early epochs rely on the labels, later epochs on M.
5. **Training:** 30 epochs, AdamW, 4-epoch warm-up to LR 0.001 then linear decay, batch 16 (by gradient accumulation).
6. **Inference:** a single network; no extra cost.

**Why PEdger rather than PEdger++:**

- **More room to improve.** PEdger leaves a 2–5 point gap to the teachers, while PEdger++ is within ~0.7 points of SAUGE on BSDS500.
- **Peer reviewed.** PEdger appeared at ACM MM 2023; PEdger++ is still a preprint.
- **Cheaper to train on a T4.** PEdger has no parameter sampling or validation-fitted ensemble.
- **A clean comparison.** PEdger++ upgrades PEdger with more *self*-teaching; our method upgrades PEdger with *external* teaching. Comparing the two answers research question 3 directly.

### 4.2 Teachers (run once, frozen; predictions cached)

| # | Teacher | Venue | Built on | BSDS500 ODS | Weights released | Role |
|---|---|---|---|---|---|---|
| 1 | **SAUGE** [18] | AAAI 2025 | SAM (frozen ViT-B) + 1.3M side network | .839 (.842 w/ VOC) | ✓ BSDS | Core teacher: SAM's object-level knowledge |
| 2 | **MuGE** [12] | CVPR 2024 | CNN (EfficientNet backbone), trained on annotator granularity | .831 single output; .850 multi-granularity (M=11) | ✓ BSDS | A different model family; its multi-granularity outputs give naturally soft maps |
| 3 | **MEMO** [20] | CVPR 2026 | DINOv2 (frozen) + masked edge prediction | .854 | **Not confirmed**: code is public, weights must be checked | Newest and strongest; very crisp edges |
| 3b | **DiffusionEdge** [14] (backup) | AAAI 2024 | Latent diffusion model | .834 | ✓ BSDS, **NYUD**, BIPED | Third teacher if MEMO weights are unavailable; the only teacher with released NYUD weights |

Code: SAUGE [Star-xing1/SAUGE](https://github.com/Star-xing1/SAUGE) · MuGE [ZhouCX117/UAED](https://github.com/ZhouCX117/UAED) · MEMO [cplusx/MEMO_Edge_Detection](https://github.com/cplusx/MEMO_Edge_Detection) · DiffusionEdge [GuHuangAI/DiffusionEdge](https://github.com/GuHuangAI/DiffusionEdge)

**Why these three:**

- **Recent.** They date from 2024 to 2026.
- **Diverse foundations** (SAM, CNN, DINOv2). Their mistakes differ, so their average should be a better teacher than any single one.
- **All clearly stronger than PEdger** on BSDS500 (.821).
- **Weights released**, or easy to check.

**Excluded:**

- **GED and EasyControlEdge:** best numbers, but no code or weights.
- **RankED:** no code repository.
- **EDMB:** code availability not confirmed.
- **UAED:** an earlier version of MuGE, so it adds little diversity.

**Caveats:**

- **DiffusionEdge outputs are crisp, almost binary** (NYUD AP only .508). Soft maps carry more information for distillation, so it may help less than its ODS suggests.
- **SAUGE, MuGE and MEMO have released weights for BSDS500 only.** On NYUDv2 and MultiCue they run zero-shot, so each teacher is measured on each dataset's validation split before use (the "teacher gate", §10).

**Teacher ensemble:** `T = mean(SAUGE, MuGE, MEMO)` of the probability maps (default). An alternative weights each teacher per pixel by its confidence `|2p − 1|` (ablated).

### 4.3 Comparison-only models (not trained by us, except the optional PEdger++ run)

- **PEdger++** [23] (716K; the main competing upgrade of PEdger)
- PiDiNet, HED, RCF, BDCN, EDTER, UAED, RankED, NBED, EDMB, EasyControlEdge, GED
- The teachers themselves

---

## 5. Current Results (the Scoreboard)

Single-scale, single-output results unless marked. Numbers come from the original papers and recent comparison tables. **Our student is in bold; the PEdger++ row is the most important comparison.**

### BSDS500: trained on BSDS + PASCAL VOC

| Model | Year | ODS | OIS | Params | FPS |
|---|---|---|---|---|---|
| PiDiNet | 2021 | .807 | .823 | 0.71M | 76 |
| **PEdger (our student)** | 2023 | **.821** | **.840** | **0.73M** | **71–92** |
| EDTER | 2022 | .832 | .847 | 468.8M | 2.1 |
| **PEdger++ (self-teaching upgrade)** | 2025 | **.835** | **.850** | 716K | 92 |
| UAED | 2023 | .838 | .855 | 72.5M | – |
| SAUGE (teacher 1) | 2025 | .842 | .862 | 1.3M trainable + frozen SAM | – |
| PEdger++Large | 2025 | .848 | .859 | 4.3M | 48.5 |
| SAUGE-L | 2025 | .849 | .869 | 2.1M trainable + frozen SAM-L | – |
| MuGE (multi-granularity, M=11) | 2024 | .855 | .860 | 93.9M | – |
| PEdger++Large w/ ResNet50 | 2025 | .857 | .861 | 12.7M | 37.6 |

PEdger's FPS is 92 in its own paper (200×200 input) and 71 in the PEdger++ paper (320×480 input). We measure it ourselves at a fixed size.

### BSDS500: trained on BSDS only

| Model | Year | ODS | OIS | AP | Params |
|---|---|---|---|---|---|
| **PEdger** | 2023 | **.813** | **.834** | – | 0.73M |
| EDTER | 2022 | .824 | .841 | .880 | 468.8M |
| UAED | 2023 | .829 | .847 | .892 | 72.5M |
| **PEdger++** | 2025 | **.830** | **.846** | – | 716K |
| MuGE (teacher 2) | 2024 | .831 | .847 | .886 | 72.5M |
| DiffusionEdge (backup teacher) | 2024 | .834 | .848 | .815 | 225M |
| EDMB | 2025 | .837 | .851 | – | 71M |
| NBED | 2024 | .838 | .849 | – | 40M |
| SAUGE (teacher 1) | 2025 | .839 | .860 | .893 | 1.3M + frozen SAM |
| SAUGE-L | 2025 | .847 | .868 | .898 | 2.1M + frozen SAM-L |
| MEMO (teacher 3) | 2026 | .854 | .861 | – | ~238M |
| EasyControlEdge | 2026 | .857 | .869 | – | FLUX-based |
| GED (claimed / re-run) | 2024 | .870 / .859 | .880 / .875 | .907 / – | SD 2.1-based |

PEdger's BSDS-only score (.813) comes from the PEdger++ paper's table.

### NYUDv2 (RGB)

| Model | ODS | OIS | AP |
|---|---|---|---|
| SAM (zero-shot) | .699 | .719 | .707 |
| PiDiNet | .733 | .747 | – |
| **PEdger (our student)** | **.742** | **.757** | – |
| BDCN | .748 | .763 | .770 |
| DiffusionEdge (NYUD weights released) | .761 | .766 | .508 |
| **PEdger++** | **.765** | **.769** | – |
| EDTER | .774 | .789 | .797 |
| RankED | .780 | .793 | .826 |
| EasyControlEdge | .788 | .797 | – |
| SAUGE-L (zero-shot, never trained on NYUD) | .794 | .803 | .813 |
| GED | .800 | .809 | .798 |

### MultiCue (ODS / OIS)

| Model | Edge | Boundary |
|---|---|---|
| PiDiNet | .855 / .860 | .818 / .830 |
| **PEdger (our student)** | **.883 / .892** | **.828 / .839** |
| EDTER | .894 / .900 | .861 / .870 |
| MEMO (teacher 3) | .894 / .900 | – |
| UAED | .895 / .902 | .864 / .872 |
| MuGE (teacher 2) | .898 / .900 | .866 – .875 (differs between papers) |
| **PEdger++** | **.901 / .903** | **.866 / .874** |
| DiffusionEdge | .904 / .909 | – |
| SAUGE (teacher 1) | .905 / .907 | not reported |
| GED | .910 / .917 | .878 / .885 |

PEdger's MultiCue numbers come from the PEdger++ paper (the original PEdger paper did not report MultiCue). Cite the original MuGE paper for its boundary score.

### Room for PEdger to learn from the teachers

| Dataset | PEdger | Best teacher available to us | Gap |
|---|---|---|---|
| BSDS500 (w/ VOC) | .821 | SAUGE .842 / MEMO .854 (BSDS only) | ~2–3 points |
| NYUDv2 | .742 | DiffusionEdge .761 (trained on NYUD); SAUGE zero-shot to be measured | ~2–5 points |
| MultiCue edge | .883 | SAUGE .905 | ~2 points |
| MultiCue boundary | .828 | MuGE .866+ | ~4 points |

---

## 6. Previous Work in Our Research Area

Only work that touches our exact topic: edge detection combined with distillation or self-teaching, annotator uncertainty, foundation models, or lightweight design.

### 6.1 Teaching lightweight edge detectors (closest to us)

| Work | What it does | How we differ |
|---|---|---|
| **PEdger** [22], ACM MM 2023 | Self-teaching: averaged copies of itself produce soft targets, trusted more as training proceeds. | **Our student.** We add external teachers, with trust set by human agreement per pixel, not by the epoch. |
| **PEdger++** [23], arXiv 2025 | More self-teaching: heterogeneous architectures, momentum copies and sampled parameters. | Self-generated knowledge only. We test whether external teaching beats or complements it. |
| **KDED** [32], PRICAI 2023 | Distillation turns labels into probability maps; 14.8M-parameter model, .832 BSDS. | Large student; no foundation-model teachers; no agreement gating. |
| **STED-KD** [33], Remote Sensing 2025 | Distils a 14M CNN edge detector into a ~1M-parameter-encoder student; .818 BSDS. | Single CNN teacher; no agreement gating; no NYUD/MultiCue. |
| **STEdge** [35], arXiv 2022 | Self-training with multi-layer teaching on unlabelled images. | Self-training, not foundation-model distillation. |

### 6.2 Using annotator disagreement in edge detection

| Work | How it uses disagreement |
|---|---|
| **RCF** [5], CVPR 2017 | Ignores pixels where only some annotators marked an edge. |
| **UAED** [11], CVPR 2023 | Models each pixel's label as a distribution; disagreement becomes learned uncertainty. |
| **MuGE** [12], CVPR 2024 | Turns agreement level into controllable edge granularity. |
| **SAUGE** [18], AAAI 2025 | Aligns SAM-based multi-granularity outputs with annotator uncertainty. |
| **RankED** [13], CVPR 2024 | Ranking losses for label uncertainty and class imbalance. |
| **PEdger / PEdger++** [22], [23] | Binarise averaged labels; otherwise discard disagreement information. |

**Key point:** All of these use disagreement to train a single model, or throw it away. None uses it to decide **where a student should trust its teachers**. That is our novelty claim, and the paper must state it precisely.

### 6.3 Foundation models for edge detection

| Work | Foundation model | Note |
|---|---|---|
| SCESAME [25], WACVW 2024 | SAM | Zero-shot edges from SAM masks. |
| Unsupervised edges via SAM [26], IEEE TII 2024 | SAM | SAM-generated labels for unsupervised training. |
| SAUGE [18], AAAI 2025 | SAM (frozen) | Teacher 1. |
| MEMO [20], CVPR 2026 | DINOv2 (frozen) | Teacher 3. |
| DiffusionEdge [14], AAAI 2024 | Diffusion model | Backup teacher. |
| GED [15] / EasyControlEdge [19] | Stable Diffusion / FLUX | Highest numbers; heavy, no code. |
| EdgeSAM (distillation) [27], arXiv 2023 | SAM → CNN | Distils SAM itself for on-device segmentation. |

### 6.4 Distillation methods we build on

| Work | What we take from it |
|---|---|
| Hinton et al. [28], 2015 | Basic soft-label distillation (our plain baseline). |
| Logit Standardization [30], CVPR 2024 | Normalise teacher and student scores before comparing them. |
| OFA-KD [29], NeurIPS 2023 | Optional transformer-to-CNN feature distillation. |
| BPKD [31], WACV 2024 | Boundaries need dedicated distillation (segmentation); related work to differentiate. |
| Canonical KD study [34], arXiv 2026 | Fair-comparison rule: plain distillation baseline and equal GPU time for all methods. |

---

## 7. The Research Gap

1. **Lightweight edge detectors learn only from themselves.** PEdger and PEdger++ build soft targets from their own predictions; neither has been taught by external, stronger models.
2. **Their trust schedule ignores where humans disagree.** Trust in the self-generated targets depends only on the epoch, is the same for every pixel, and the annotator-agreement information is binarised away.
3. **No work distils recent foundation-model edge detectors into a sub-1M-parameter detector**, and **none uses annotator agreement to steer distillation**, single- or multi-teacher (§6.1–6.2).
4. **No controlled comparison exists** between upgrading a compact detector with more self-teaching (PEdger → PEdger++) and upgrading it with external teaching.
5. **Missing numbers:** base SAUGE, MuGE and MEMO on NYUDv2 (zero-shot) and MultiCue boundary; PEdger AP scores and crispness-aware scores.

---

## 8. Our Method

### 8.1 Overview

```
        once, offline                                   every training step (inside PEdger)
┌───────────────────────────────────┐      ┌──────────────────────────────────────────────────┐
│ SAUGE ─┐                          │      │ image, GT Y, teacher map T, agreement a           │
│ MuGE  ─┼─► mean ─► teacher map T  │─────►│   ─► same random crop/flip/rotate/scale          │
│ MEMO  ─┘   saved to disk          │      │   ─► PEdger forward (recurrent + non-recurrent)  │
└───────────────────────────────────┘      │   ─► soft target from Y, self-ensemble M and T,  │
                                           │      weighted per pixel by human agreement       │
                                           └──────────────────────────────────────────────────┘
Inference: unchanged PEdger (0.73M params, single network)
```

### 8.2 Agreement map (BSDS500)

For pixel *p*, let `a(p)` ∈ [0, 1] be the fraction of annotators who marked it as an edge (the averaged GT PNG).

- Agreement: `agree(p) = |2·a(p) − 1|`. This is 1 when all annotators agree and 0 when they are split evenly.
- Human-trust weight: `α(p) = α_min + (1 − α_min) · agree(p)`.
- Optional 1–2 px dilation of each annotator's map before averaging, because hand-drawn lines are often slightly offset (ablated).

### 8.3 Two ways to plug the teachers into PEdger

**Variant A: target-level (main method).** Replace PEdger's epoch-based soft target with a per-pixel, agreement-aware one:

```
PEdger original:  Ỹ(p) = η_ep · M(p) + (1 − η_ep) · Y(p)

Ours:             Ỹ(p) = α(p) · Y(p) + (1 − α(p)) · [ (1 − γ) · M(p) + γ · T(p) ]
```

- **Where humans agree** (α high), the target stays close to the human label.
- **Where humans disagree** (α low), the target comes from the models: PEdger's own ensemble M and the external teacher ensemble T, mixed by γ.
- γ is constant (ablated: 0, 0.5, 1) or set per pixel by relative confidence, `γ(p) ∝ |2T(p) − 1| / (|2T(p) − 1| + |2M(p) − 1|)`.
- To keep PEdger's warm-up behaviour, (1 − α(p)) is multiplied by the ramp `ep / EP`, so early epochs still rely on the labels (ablated).
- **Sanity check:** γ = 0 with α(p) = 1 − η_ep recovers the original PEdger exactly.

**Variant B: loss-level (simpler baseline).** Keep PEdger untouched and add a separate distillation term:

```
L = L_PEdger + λ_kd · Σ_p w(p) · BCE( σ(z_S(p)/τ), σ(z_T(p)/τ) ) / Σ_p w(p),     w(p) = 1 − α(p) + w_min
```

This uses logit standardization adapted from [30] (per-image z-scoring of logits before the temperature τ; our adaptation, ablated). Variant A must beat Variant B to justify its extra complexity.

### 8.4 Multi-teacher ensemble

- **Default:** `T(p) = mean(T_SAUGE(p), T_MuGE(p), T_MEMO(p))`.
- **Alternative (ablated):** confidence-weighted, `T(p) = Σ_k c_k(p) · T_k(p) / Σ_k c_k(p)` with `c_k(p) = |2T_k(p) − 1|`.
- For MuGE, use the average of its granularity outputs as its soft map.
- If a teacher fails the teacher gate on a dataset (not better than PEdger on that validation split), drop it from the ensemble for that dataset.

### 8.5 NYUDv2 and MultiCue

NYUDv2 has one annotation per image, so there is no agreement map. We test:

- **(a)** a constant α following PEdger's η schedule;
- **(b)** a teacher-confidence weight, `α(p) = 1 − c·|2T(p) − 1|`.

On NYUDv2, the teacher ensemble includes DiffusionEdge's released NYUD model if it passes the gate. MultiCue has multiple annotations, so the BSDS approach applies (verify when downloading).

### 8.6 Optional final experiment: PEdger++ + our method

PEdger++ uses the same soft-target form (`Ỹ = η·M + (1−η)·Y`), so Variant A plugs in the same way. Use the "efficient" PEdger++ training to save GPU time. A gain here shows external teaching and richer self-teaching add up.

---

## 9. Hypotheses and Expected Results

| ID | Hypothesis | How we test it | What would count as support |
|---|---|---|---|
| **H1** | External teachers improve PEdger at zero extra inference cost. | Our retrained PEdger vs PEdger + distillation (Variant B, uniform weight, SAUGE only), same GPU time, 3 seeds. | ODS gain beyond seed-to-seed variation (aim ≥ +0.5 on BSDS500); params and FPS unchanged. |
| **H2** | Agreement-aware weighting beats uniform weighting. | Uniform vs agreement-weighted (Variants A and B). | Higher ODS/OIS; **gains concentrated on low-agreement pixels** (agreement-bin analysis below). |
| **H3** | A multi-teacher ensemble beats the best single teacher. | SAUGE, MuGE, MEMO alone vs their ensemble. | Ensemble-taught student has the highest ODS. |
| **H4** | External teaching is at least as effective as richer self-teaching. | PEdger + ours vs PEdger++ (published and retrained), with training GPU-hours reported. | PEdger + ours ≥ PEdger++ ODS, or close at lower training cost. |
| **H5** | External teaching and self-teaching add up. | PEdger++ vs PEdger++ + ours (optional, §8.6). | Any consistent gain over PEdger++. |
| **H6** | Gains are largest where the teacher–student gap is largest. | Per-dataset gains vs the measured teacher–student gap on each validation set. | Largest gains on NYUDv2 and MultiCue boundary. |

**Agreement-bin analysis (new):** split BSDS500 test edge pixels into low / medium / high human agreement, and report precision and recall per bin for PEdger, PEdger++ and our model. If H2 holds, the improvement should appear mainly in the low-agreement bin.

**Realistic expectations (hypotheses, not promises):**

| Dataset | PEdger (published) | Realistic aim | Ambitious headline |
|---|---|---|---|
| BSDS500 (w/ VOC) | .821 / .840 | .828 – .834 | ≥ .835, matching or beating PEdger++ |
| NYUDv2 | .742 / .757 | .750 – .760 | ≥ .765, matching PEdger++ |
| MultiCue boundary | .828 / .839 | .840 – .855 | ≥ .866, matching PEdger++ |

If PEdger + ours falls short of PEdger++, the paper still contributes:

- the gain over PEdger itself (H1–H3);
- the agreement-bin analysis;
- the self- vs external-teaching comparison including training cost;
- the optional PEdger++ + ours result (H5).

---

## 10. What We Have to Code

Everything runs on Kaggle (T4, 16 GB, ~12 h per session, roughly 30 GPU-h per week). Use PyTorch mixed precision (`torch.cuda.amp`), and save checkpoints every epoch. We **build on the official PEdger code** rather than rewriting it.

### 10.1 Repository layout

```
pedger-mtkd/
├── third_party/
│   ├── PEdger/                  # official PEdger code (unmodified submodule)
│   └── PEdgerPlus/              # official PEdger++ code (optional final experiment)
├── configs/                     # YAML per experiment: baseline, B_uniform, B_agree, A_*, single/ensemble teachers
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
│   ├── soft_target.py           # Variant A: agreement-aware soft target with M, T, γ, ramp
│   ├── kd_loss.py               # Variant B: weighted KD loss + logit standardization
│   └── train_kd.py              # PEdger training loop with minimal hooks for A/B
├── eval/
│   ├── run_bsds_eval.py         # NMS + ODS/OIS/AP (MATLAB or verified Python port)
│   └── agreement_bins.py        # per-agreement-bin precision/recall (H2)
├── bench/speed_flops.py         # params, FLOPs, FPS (T4 + CPU), training GPU-hours from logs
└── notebooks/kaggle_*.ipynb     # one notebook per stage; data and caches stored as Kaggle Datasets
```

### 10.2 Coding tasks in order

| # | Task | Details | Done when |
|---|---|---|---|
| 1 | **Evaluation first** | Set up the BSDS benchmark; evaluate a released PiDiNet or PEdger checkpoint. | You reproduce the published ODS within ±0.002. |
| 2 | **Data preparation** | Same train lists and augmentation as PEdger; NYUD; MultiCue 3 splits. | Loaders return correct images and labels (visual check). |
| 3 | **Agreement maps** | `agreement.py` from the averaged GT; with and without dilation. | Maps look right on 10 sample images. |
| 4 | **Teacher caching + teacher gate** | Check MEMO weights (else DiffusionEdge). Cache SAUGE, MuGE and MEMO maps on original training images. Score each teacher and the ensemble on each validation split. | Teacher ODS table per dataset. Teachers not better than PEdger on a dataset are dropped for that dataset. |
| 5 | **PEdger baseline reproduction** | Train official PEdger in our pipeline (BSDS+VOC). | Within ~0.005 of .821 ODS. Otherwise our retrained number is the baseline (§12). |
| 6 | **Joint augmentation** | Extend the PEdger dataset so T and a(p) receive the same geometric transforms as the image and GT. | Overlay check: the teacher map stays aligned after augmentation. |
| 7 | **Variant B (loss-level)** | `kd_loss.py`: weighted soft-BCE, temperature, logit standardization, `w_min`. | Unit test: with λ_kd = 0, training is identical to the baseline. |
| 8 | **Variant A (target-level)** | `soft_target.py`: replace `Ỹ = η·M + (1−η)·Y` with the agreement-aware target; constant and confidence-based γ; optional ramp. | Unit test: γ = 0 and α = 1 − η reproduce the original target exactly. |
| 9 | **Training runs** | Baseline; B-uniform; B-agree; A-agree with each single teacher and the ensemble; γ sweep; 3 seeds for main results. | All runs logged with GPU-hours. |
| 10 | **Evaluation and analysis** | ODS/OIS/AP (SS/MS, ± VOC); NYUD; MultiCue; agreement bins; crispness (optional). | Results tables filled. |
| 11 | **Efficiency and cost** | Params, FLOPs, FPS for PEdger and ours (identical); training GPU-hours for PEdger, ours and PEdger++. | Efficiency and training-cost table. |
| 12 | **Optional: PEdger++ + ours** | Plug Variant A into the efficient PEdger++. | One result row for H5. |

### 10.3 Core pieces (sketch)

```python
import torch, torch.nn.functional as F

def human_trust(a, a_min=0.1):               # a = fraction of annotators marking edge
    agree = (2 * a - 1).abs()                # 1 = everyone agrees, 0 = evenly split
    return a_min + (1 - a_min) * agree       # α(p)

def teacher_ensemble(maps, weighted=False):  # maps: list of (B,1,H,W) teacher probabilities
    T = torch.stack(maps)
    if not weighted:
        return T.mean(0)
    c = (2 * T - 1).abs() + 1e-6             # per-pixel confidence of each teacher
    return (c * T).sum(0) / c.sum(0)

# Variant A: replaces PEdger's  Y_soft = eta*M + (1-eta)*Y
def soft_target(Y, M, T, alpha, gamma=0.5, ramp=1.0):
    model_view = (1 - gamma) * M + gamma * T          # self-ensemble + external teachers
    trust_models = (1 - alpha) * ramp                 # ramp = ep/EP keeps early epochs label-driven
    return (1 - trust_models) * Y + trust_models * model_view

# Variant B: extra loss on top of the unchanged PEdger loss
def standardize(z, eps=1e-6):
    return (z - z.mean((2, 3), keepdim=True)) / (z.std((2, 3), keepdim=True) + eps)

def kd_loss(z_s, p_t, w, tau=2.0, use_ls=True):
    z_t = torch.logit(p_t.clamp(1e-4, 1 - 1e-4))
    if use_ls:
        z_s, z_t = standardize(z_s), standardize(z_t)
    loss = F.binary_cross_entropy_with_logits(z_s / tau, torch.sigmoid(z_t / tau), reduction="none")
    return (w * loss).sum() / w.sum().clamp_min(1.0)
```

---

## 11. Experiments and Timeline

### Main results tables (what the paper will show)

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
| SOTA comparison rows | from §5 | from §5 | from §5 |

### Ablations (BSDS500)

- Teacher choice: SAUGE / MuGE / MEMO / mean ensemble / confidence-weighted ensemble.
- α_min; dilation of agreement maps.
- γ (0 / 0.5 / 1 / confidence-based); keeping or dropping the ep/EP ramp.
- λ_kd and τ (Variant B); logit standardization on/off.

### Five-week plan

| Week | Work |
|---|---|
| 1 | Evaluation reproduced (task 1); data ready (tasks 2–3); check MEMO weights; cache all teachers and run the teacher gate (task 4). |
| 2 | PEdger baseline reproduced (task 5); joint augmentation (task 6); Variants B and A implemented and unit-tested (tasks 7–8). |
| 3 | BSDS500: single-teacher and ensemble runs, Variants A/B (task 9). |
| 4 | Ablations; final 3-seed runs; NYUDv2 and MultiCue (3 splits) (tasks 9–10). |
| 5 | Agreement-bin analysis; efficiency and training cost (task 11); optional PEdger++ + ours (task 12); figures and writing. |

**GPU budget:** time the first full PEdger run before planning the rest. If one run takes T hours, the weekly quota allows about 30/T runs. Prioritise main results with 3 seeds over wide hyperparameter sweeps; run sweeps with one seed.

---

## 12. Risks and Fallbacks

| Risk | Fallback |
|---|---|
| MEMO weights not released | Use DiffusionEdge as the third teacher (weights for BSDS and NYUD). |
| A teacher is weak zero-shot on NYUDv2 or MultiCue | Teacher gate drops it for that dataset; on NYUD rely on DiffusionEdge's NYUD model and/or fine-tune SAUGE's 1.3M side network (SAM frozen). |
| DiffusionEdge maps are too binary to be useful soft labels | Lightly blur its maps (e.g., a small Gaussian blur) before use, or lower its ensemble weight; report the effect. |
| Our PEdger retrain falls below .821 | Report our retrained number as the baseline and compare fairly against it; state this openly. |
| PEdger + ours does not beat PEdger++ | Report honestly; lead with H1–H3, the agreement-bin analysis and the training-cost comparison; add the optional PEdger++ + ours result (H5). |
| Python evaluation differs from MATLAB | Validate on released checkpoints first (task 1). |
| Kaggle quota or session limits | Checkpoint every epoch; store cached teacher maps and datasets as Kaggle Datasets; one seed for ablations. |
| A new paper does the same thing | Re-check Google Scholar and arXiv right before submission. |

---

## 13. References

| # | Paper | Venue | Link |
|---|---|---|---|
| 1 | Contour Detection and Hierarchical Image Segmentation (BSDS500) | IEEE TPAMI 2011 | [author PDF](https://vision.ics.uci.edu/papers/ArbelaezMFM_PAMI_2011/ArbelaezMFM_PAMI_2011.pdf) |
| 2 | Indoor Segmentation and Support Inference from RGBD Images (NYUDv2) | ECCV 2012 | [author PDF](https://cs.nyu.edu/~fergus/datasets/indoor_seg_support.pdf) |
| 3 | A Systematic Comparison between Visual Cues for Boundary Detection (MultiCue) | Vision Research 2016 | [author PDF](https://serre-lab.clps.brown.edu/wp-content/uploads/2016/11/2016_Mely_VR.pdf) |
| 4 | Holistically-Nested Edge Detection (HED) | ICCV 2015 | [arXiv](https://arxiv.org/abs/1504.06375) |
| 5 | Richer Convolutional Features for Edge Detection (RCF) | CVPR 2017 | [arXiv](https://arxiv.org/abs/1612.02103) |
| 6 | Bi-Directional Cascade Network for Perceptual Edge Detection (BDCN) | CVPR 2019 | [arXiv](https://arxiv.org/abs/1902.10903) |
| 7 | Dense Extreme Inception Network (DexiNed) | WACV 2020 | [arXiv](https://arxiv.org/abs/1909.01955) |
| 8 | Pixel Difference Networks for Efficient Edge Detection (PiDiNet) | ICCV 2021 | [arXiv](https://arxiv.org/abs/2108.07009) |
| 9 | EDTER: Edge Detection with Transformer | CVPR 2022 | [arXiv](https://arxiv.org/abs/2203.08566) |
| 10 | Segment Anything (SAM) | ICCV 2023 | [arXiv](https://arxiv.org/abs/2304.02643) |
| 11 | The Treasure Beneath Multiple Annotations: An Uncertainty-aware Edge Detector (UAED) | CVPR 2023 | [arXiv](https://arxiv.org/abs/2303.11828) |
| 12 | **MuGE: Multiple Granularity Edge Detection** (teacher 2) | CVPR 2024 | [CVF PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_MuGE_Multiple_Granularity_Edge_Detection_CVPR_2024_paper.pdf) |
| 13 | RankED: Ranking-based Losses for Edge Detection | CVPR 2024 | [arXiv](https://arxiv.org/abs/2403.01795) |
| 14 | **DiffusionEdge: Diffusion Probabilistic Model for Crisp Edge Detection** (backup teacher) | AAAI 2024 | [arXiv](https://arxiv.org/abs/2401.02032) |
| 15 | Generative Edge Detection with Stable Diffusion (GED) | arXiv 2024 (preprint) | [arXiv](https://arxiv.org/abs/2410.03080) |
| 16 | A New Baseline for Edge Detection (NBED) | arXiv 2024 (preprint) | [arXiv](https://arxiv.org/abs/2409.14976) |
| 17 | EDMB: Edge Detector with Mamba | WACV 2025 | [arXiv](https://arxiv.org/abs/2501.04846) |
| 18 | **SAUGE: Taming SAM for Uncertainty-Aligned Multi-Granularity Edge Detection** (teacher 1) | AAAI 2025 | [arXiv](https://arxiv.org/abs/2412.12892) |
| 19 | EasyControlEdge: A Foundation-Model Fine-Tuning for Edge Detection | arXiv 2026 (preprint) | [arXiv](https://arxiv.org/abs/2602.16238) |
| 20 | **MEMO: Human-like Crisp Edge Detection Using Masked Edge Prediction** (teacher 3) | CVPR 2026 | [arXiv](https://arxiv.org/abs/2603.20782) |
| 21 | MS2Edge: Energy-Efficient and Crisp Edge Detection in SNNs | arXiv 2025 (preprint) | [arXiv](https://arxiv.org/abs/2511.13735) |
| 22 | **Practical Edge Detection via Robust Collaborative Learning (PEdger)** (our student) | ACM MM 2023 | [arXiv](https://arxiv.org/abs/2308.14084) · [code](https://github.com/ForawardStar/PEdger) |
| 23 | **PEdger++: Practical Edge Detection via Assembling Cross Information** (main comparison) | arXiv 2025 (preprint) | [arXiv](https://arxiv.org/abs/2508.11961) · [code](https://github.com/ForawardStar/EdgeDetectionviaPEdgerPlus/) |
| 24 | Tiny and Efficient Model for the Edge Detection Generalization (TEED) | ICCV Workshops 2023 | [arXiv](https://arxiv.org/abs/2308.06468) |
| 25 | Zero-Shot Edge Detection with SCESAME | WACV Workshops 2024 | [arXiv](https://arxiv.org/abs/2308.13779) |
| 26 | Boosting Deep Unsupervised Edge Detection via Segment Anything Model | IEEE TII 2024 | [IEEE Xplore](https://ieeexplore.ieee.org/document/10490131) (paywalled) |
| 27 | EdgeSAM: Prompt-In-the-Loop Distillation for On-Device SAM | arXiv 2023 (preprint) | [arXiv](https://arxiv.org/abs/2312.06660) |
| 28 | Distilling the Knowledge in a Neural Network | NIPS DL Workshop 2014 | [arXiv](https://arxiv.org/abs/1503.02531) |
| 29 | One-for-All: Heterogeneous Knowledge Distillation (OFA-KD) | NeurIPS 2023 | [arXiv](https://arxiv.org/abs/2310.19444) |
| 30 | Logit Standardization in Knowledge Distillation | CVPR 2024 | [arXiv](https://arxiv.org/abs/2403.01427) |
| 31 | BPKD: Boundary Privileged Knowledge Distillation for Semantic Segmentation | WACV 2024 | [arXiv](https://arxiv.org/abs/2306.08075) |
| 32 | KDED: A Knowledge Distillation Based Edge Detector | PRICAI 2023 | [Springer](https://link.springer.com/chapter/10.1007/978-981-99-7025-4_11) (paywalled) |
| 33 | Satellite Optical Target Edge Detection Based on Knowledge Distillation (STED-KD) | Remote Sensing 2025 | [MDPI](https://www.mdpi.com/2072-4292/17/17/3008) |
| 34 | The Surprising Effectiveness of Canonical KD for Semantic Segmentation | arXiv 2026 (preprint) | [arXiv](https://arxiv.org/abs/2604.25530) |
| 35 | STEdge: Self-training Edge Detection with Multi-layer Teaching and Regularization | arXiv 2022 (preprint) | [arXiv](https://arxiv.org/abs/2201.05121) |

---

*Before submission: re-check every number against the original paper, confirm the annotation format of each downloaded dataset, and repeat the literature search for newer work on distillation for edge detection.*
