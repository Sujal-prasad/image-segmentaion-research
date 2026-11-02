# Requirements

What the project must deliver and the constraints it works under. Derived from [Edge_Detection_Research_Plan.md](../Edge_Detection_Research_Plan.md).

## 1. Goal

Improve PEdger's edge detection accuracy on BSDS500, NYUDv2 and MultiCue by distilling from frozen foundation-model edge detectors, steered per pixel by human annotator agreement, **without changing PEdger's size or inference cost**.

## 2. Functional requirements

| ID | Requirement | Acceptance |
|---|---|---|
| FR-1 | **Evaluation pipeline**: NMS + ODS / OIS / AP, single- and multi-scale, BSDS-standard matching. | Reproduces a released PiDiNet or PEdger checkpoint's published ODS within **±0.002**. |
| FR-2 | **Data preparation** for BSDS500 (+ PASCAL VOC aug), NYUDv2 and MultiCue, using PEdger's exact lists and augmentation. MultiCue: 3 random splits with fixed seeds. | Loaders return correct image/label pairs (visual check). |
| FR-3 | **Agreement maps** `a(p)`, `agree(p)`, `α(p)` from averaged GT, with optional 1–2 px dilation. | Maps look correct on 10 sample images. |
| FR-4 | **Teacher caching**: run SAUGE, MuGE, MEMO (or DiffusionEdge) once on original training images; save 16-bit PNG probability maps. | Cache complete for every training image of each dataset. |
| FR-5 | **Teacher ensemble**: mean (default) and confidence-weighted. | Unit test on synthetic maps. |
| FR-6 | **Teacher gate**: score each teacher and the ensemble on each dataset's validation split vs PEdger; drop weaker teachers per dataset. | Teacher ODS table per dataset. |
| FR-7 | **PEdger baseline** retrained in our pipeline (BSDS + VOC). | Within ~0.005 of .821 ODS, or our number becomes the reported baseline. |
| FR-8 | **Joint augmentation**: image, `Y`, `T`, `a(p)` share one geometric transform. | Overlay check passes. |
| FR-9 | **Variant B** (loss-level KD): weighted soft-BCE, temperature `τ`, optional logit standardization, `w_min`, `λ_kd`. | With `λ_kd = 0`, training is identical to baseline. |
| FR-10 | **Variant A** (target-level KD): agreement-aware soft target; constant and confidence-based `γ`; optional `ep/EP` ramp. | With `γ = 0`, `α = 1 − η`, target equals PEdger's exactly. |
| FR-11 | **NYUDv2 α options**: constant schedule and teacher-confidence weighting. | Both selectable by config. |
| FR-12 | **Experiment configs**: every run defined by one YAML file. | Any reported number can be regenerated from its config + seed. |
| FR-13 | **Agreement-bin analysis**: precision / recall per low / medium / high agreement bin on BSDS500 test, for PEdger, PEdger++ and ours. | Table produced. |
| FR-14 | **Efficiency report**: params, FLOPs, FPS (T4 + CPU, batch 1, after warm-up, fixed input e.g. 320×480), training GPU-hours. | Table produced; student numbers equal PEdger's. |
| FR-15 | *(Optional)* **PEdger++ + ours**: Variant A plugged into efficient PEdger++. | One result row for H5. |
| FR-16 | *(Optional)* Crispness-aware evaluation (CEval / AC). | Reported if time allows. |

## 3. Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-1 | **Zero inference overhead**: student params, FLOPs and FPS identical to PEdger (0.73M params). |
| NFR-2 | **Compute budget**: Kaggle T4, 16 GB VRAM, ~12 h per session, ~30 GPU-h per week. |
| NFR-3 | **Resumability**: full training state checkpointed every epoch and every ~20 min; clean save-and-exit before the session limit; automatic resume in a new Kaggle session with no code or config change; a resumed run matches an uninterrupted one. Teacher caching and evaluation also resume. GPU-hours accumulate across sessions. |
| NFR-4 | **Mixed precision** (`torch.cuda.amp`) for training. |
| NFR-5 | **Reproducibility**: fixed seeds; config, code commit and seed logged for every run. |
| NFR-6 | **Fair comparison**: plain-distillation baseline included; equal GPU time across compared methods; training GPU-hours reported (needed vs PEdger++). |
| NFR-7 | **Statistical care**: 3 seeds for main results; MultiCue reported as mean ± std over 3 splits. |
| NFR-8 | **Reporting hygiene**: never mix SS / MS or ± VOC in one comparison column. |
| NFR-9 | **Non-invasive**: official PEdger / PEdger++ code stays unmodified. |

## 4. Datasets

| Dataset | Split | Annotations | Role |
|---|---|---|---|
| BSDS500 | 200 train, 100 val, 200 test; train on train+val (augmented) | ~5 annotators per image (averaged GT) | Main benchmark; agreement maps |
| PASCAL VOC Context (aug) | ~10k images, training only | Single | Extra training data |
| NYUDv2 | 381 train, 414 val, 654 test; train on 795 | Single | Second benchmark |
| MultiCue | 80 train / 20 test, 3 random splits | Edge + boundary labels, several annotations (count to verify) | Third benchmark |

## 5. Evaluation protocol

- NMS, then matching against ground truth.
- `maxDist`: **0.0075** for BSDS500 and MultiCue, **0.011** for NYUDv2.
- Tool: MATLAB BSDS / edges-toolbox; Python port (e.g. pyEdgeEval) only after it reproduces a published number.
- Metrics: **ODS** (headline), OIS, AP, Params / FLOPs / FPS, training GPU-hours; optional CEval / AC.

## 6. Software and hardware dependencies

| Dependency | Purpose |
|---|---|
| Python 3.x, PyTorch (CUDA, AMP) | Training and inference |
| torchvision, OpenCV and/or Pillow, NumPy | Images, augmentation, 16-bit PNG I/O |
| PyYAML | Experiment configs |
| `fvcore` or `thop` | FLOP counting |
| MATLAB BSDS / edges toolbox (or validated pyEdgeEval) | Evaluation |
| Official repos: PEdger, PEdger++, SAUGE, MuGE (UAED repo), MEMO, DiffusionEdge | Student baseline, teachers |
| Kaggle (T4 GPU, Kaggle Datasets) | Compute and storage |

Exact versions are pinned once the PEdger baseline runs (a `requirements.txt` / environment file will follow the official PEdger environment).

## 7. Acceptance criteria (hypotheses)

| ID | Hypothesis | Supported if |
|---|---|---|
| H1 | External teachers improve PEdger at zero extra inference cost. | ODS gain beyond seed variation (aim ≥ +0.5 on BSDS500); params and FPS unchanged. |
| H2 | Agreement-aware weighting beats uniform weighting. | Higher ODS/OIS; gains concentrated in the low-agreement bin. |
| H3 | Multi-teacher ensemble beats the best single teacher. | Ensemble-taught student has the highest ODS. |
| H4 | External teaching ≥ richer self-teaching. | PEdger + ours ≥ PEdger++ ODS, or close at lower training cost. |
| H5 | External teaching and self-teaching add up. *(optional)* | Consistent gain of PEdger++ + ours over PEdger++. |
| H6 | Gains are largest where the teacher–student gap is largest. | Largest gains on NYUDv2 and MultiCue boundary. |

## 8. Out of scope

- Training or fine-tuning the teachers (except, as a fallback, SAUGE's 1.3M side network on NYUD with SAM frozen).
- New student architectures.
- Models without released code/weights (GED, EasyControlEdge, RankED, EDMB) except as published comparison rows.
