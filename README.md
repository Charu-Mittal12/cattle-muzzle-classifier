# Attention-Based Cattle Identification from Muzzle Images
 
Reimplementation and extension of **Pathak & Prakash (2025), "Attention-based
multi-modal robust cattle identification technique using deep learning,"**
*Computers and Electronics in Agriculture* 238:110747.
 
Identifies individual cattle from muzzle images (unique bead/ridge patterns, similar
to a human fingerprint) using a CNN backbone + custom spatial attention module. This
is a non-invasive alternative to ear tags, RFID, and branding for livestock ID and
insurance/ownership verification.
 
## What this repo actually does
 
Everything below was run and produces the outputs described — no placeholder claims.
 
1. **Muzzle classifier (main pipeline)** — EfficientNetV2B0 backbone (ImageNet
   pretrained) + a custom spatial attention module (Eq. 7–9 of the paper) +
   classification head, trained on **568 cattle identities** with a two-phase
   transfer-learning recipe (frozen backbone → fine-tune last 3 layers).
2. **Backbone comparison** — the same architecture pattern re-run with VGG16,
   ResNet50, and EfficientNetV2B0, compared on accuracy, macro/micro/weighted F1,
   parameter count, and FLOPs.
3. **Spatial attention ablation** — EfficientNetV2B0 trained with and without the
   spatial attention module, same data/schedule, to directly test whether attention
   is actually helping.
4. **Evaluation tooling** — bootstrap 95% confidence intervals, normalized confusion
   matrix, Grad-CAM visualizations, and an "Algorithm 1" score-fusion function for
   multi-modal (muzzle + face) prediction.
## Dataset
 
- **CMPD568** = CMPD300 (Shojaeipour et al., 2021) ∪ CMPD268 (Li et al., 2022a),
  muzzle-only crops, 568 classes, 7,555 images total.
- Split used: 4,609 train / 1,333 val / 1,613 test.
- Images are **pre-cropped** (no detection step run in this repo — see Limitations).
## Results
 
### Main run — EfficientNetV2B0 + Spatial Attention
| Metric | Value |
|---|---|
| Test accuracy | 98.2–98.7% |
| Macro-F1 | ~0.97 |
| Weighted-F1 | ~0.98 |
 
### Backbone comparison (15+15 epoch schedule, CMPD568-muzzle)
| Backbone | Test Acc | Macro-F1 | Params (M) | FLOPs (GB) |
|---|---|---|---|---|
| **VGG16** | **98.70%** | **0.9816** | 15.01 | 30.71 |
| ResNet50 | 96.90% | 0.9521 | 24.76 | 7.75 |
| EfficientNetV2B0 | 97.58% | 0.9618 | 6.65 | 1.46 |
 
VGG16 edged out the other two on accuracy in this run, but EfficientNetV2B0 uses
~21x fewer FLOPs for a ~1-point accuracy tradeoff — consistent with the paper's own
framing of EfficientNetV2B0 as the efficiency pick rather than the outright top
scorer. **Latency numbers were measured but are not reliable** (batch=1 timing
artifact — EfficientNetV2B0 showed higher latency than ResNet50 despite far fewer
FLOPs) and are excluded from the table above; see Limitations.
 
### Spatial attention ablation (EfficientNetV2B0, same data/schedule)
| Variant | Macro-F1 |
|---|---|
| No spatial attention | 0.9598 |
| With spatial attention | 0.9582 |
 
Spatial attention changed macro-F1 by **-0.16%** in this run — within noise for a
15-epoch schedule. This doesn't confirm the paper's attention module is providing a
measurable benefit at this training budget; a longer schedule or repeated seeds would
be needed to say more.
 
## What's in the notebook
 
Single Kaggle notebook, structured as sequential cells:
 
| Cells | Content |
|---|---|
| 1–5 | Data pipeline (`tf.data`), spatial attention module, model builder |
| 6–7 | Two-phase training (frozen → fine-tuned) |
| 8–12 | Training curves, test evaluation, bootstrap CI, confusion matrix, Grad-CAM |
| 13 | Ablation helper (fine-tune depth × attention on/off) |
| 14 | Multi-modal score fusion (Algorithm 1) — requires a separately trained face model |
| 16–17 | Backbone comparison (VGG16 / ResNet50 / EfficientNetV2B0) |
| 18–21 | Class-imbalance experiment on CMPD268 (focal loss vs weighted CE) — **not run**, see below |
| 22 | Spatial attention ablation |
 
## Known limitations / what's not done
 
- **No object detection step.** The paper trains a YOLOv8n detector to crop
  face/muzzle from raw images; this repo skips that because the released dataset is
  already cropped. The pipeline assumes pre-cropped input.
- **Class-imbalance experiment (Cells 18–21) did not execute** — the code (baseline
  cross-entropy vs class-weighted CE vs focal loss vs focal+weighted) is written but
  `CFG268.DATA_ROOT` was never pointed at a real CMPD268 dataset path, so this cell
  exits early with a warning. No imbalance results exist yet.
- **Latency benchmarking is unreliable** as noted above — needs larger batch size and
  more warmup iterations before it can be trusted or reported.
- **No face-feature model trained**, so the Algorithm 1 multi-modal fusion function
  exists but has never been run end-to-end.
- Dataset was captured in controlled daylight with burst-mode photography, which
  raises a real risk of near-duplicate frames leaking across train/val/test — no
  perceptual-hash leakage check has been done, so these accuracy numbers should be
  read as optimistic upper bounds rather than confirmed generalization performance.
- Muzzle biometrics are not an established legal standard for proof of ownership in
  most jurisdictions.
## Engineering deviations from the paper (for the record)
 
| Paper | This repo | Why |
|---|---|---|
| Batch size 10 | 64 | Paper used a CPU-only box; this maximizes GPU throughput |
| Single run, ≤100 epochs | Explicit Phase 1 + Phase 2 | Standard staged transfer learning |
| Rotation ±15° only | + horizontal flip | Mild extra augmentation |
| — | Mixed precision (fp16) + XLA JIT | Faster training on Kaggle T4 GPUs |
 
## Reference
 
Pathak, P., & Prakash, S. (2025). Attention-based multi-modal robust cattle
identification technique using deep learning. *Computers and Electronics in
Agriculture*, 238, 110747.
 




