# Open-Set Deepfake Detection with SigLIP 2

A production-ready, research-oriented re-implementation of

> **Open-Set Deepfake Detection: A Parameter-Efficient Adaptation Method with
> Forgery Style Mixture** (Kong et al., submitted to IEEE T-CSVT; arXiv:2408.12791)

with the paper's ViT/CLIP backbone replaced by **SigLIP 2**
(`google/siglip2-base-patch16-224`) while preserving the methodology.

The method (**OSDFD**) adapts a **frozen** vision transformer to open-set face
forgery detection by training only lightweight, forgery-aware modules:

| Component | Where | Paper |
|---|---|---|
| **LoRA** | attention Q/K/V projections | Sec. III-B, Eqs. 4–5 |
| **CDC Adapter** (Central Difference Convolution) | FFN blocks | Sec. III-B, Eqs. 1–3, Fig. 5 |
| **Forgery Style Mixture (FSM)** | patch tokens, train-only, fake samples | Sec. III-C, Eqs. 6–9 |
| **Single-Center Loss (SCL)** | penultimate head feature | Sec. III-D, Eqs. 11–13 |
| Objective | `L = BCE + λ·SCL` (λ=1) | Sec. III-D, Eq. 10 |

Only the LoRA layers, the CDC adapter and the classification head are optimised;
the SigLIP 2 weights stay frozen (~1–3M trainable parameters).

---

## Why SigLIP 2

The paper freezes an ImageNet/CLIP ViT and preserves its pre-trained priors. We
swap in a SigLIP 2 vision tower, which follows the standard ViT architecture
with a **MAP (Multihead Attention Pooling)** head and **no `[CLS]` token**
(SigLIP 2 paper, Sec. 2.1). The fixed-resolution `siglip2-base-patch16-224`
checkpoint is backward compatible with the SigLIP v1 architecture, so it takes
standard `(B, 3, 224, 224)` inputs and yields **196 patch tokens** on a 14×14
grid — convenient for the CDC adapter's token→grid reshape.

- **Global feature**: MAP-pooled embedding.
- **Local features**: last-layer patch tokens (used by CDC, FSM and optional
  global+local fusion).

---

## Project structure

```
configs/            # Hydra config groups (backbone, peft, fsm, loss, ...)
docs/               # the two source papers
scripts/            # train.sh / eval.sh / infer.sh (activate the conda env)
  preprocess_ffpp.py  # FF++ videos -> face crops
  download/         # dataset download helpers (HuggingFace Hub)
src/
  data/             # dataset, LightningDataModule, transforms, face detection
  models/           # SigLIP2 backbone, LoRA, CDC adapter, FSM, head, OSDFD model
  losses/           # Single-Center Loss + combined objective
  lightning/        # LightningModule (train/val/test/predict steps)
  training/         # metrics (ACC/AUC/F1/AP/EER/FPR/FNR) + figures
  inference/        # standalone predictor
  utils/            # seeding + Lightning factory helpers
tests/              # fast offline smoke tests
train.py test.py predict.py
requirements.txt
```

---

## Installation

All work uses the pre-existing conda environment **`open_set_deepfake`**.

```bash
conda activate open_set_deepfake

# PyTorch matching your CUDA (example: CUDA 12.1):
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Everything else:
pip install -r requirements.txt
```

> SigLIP 2 requires `transformers>=4.53`. Optional face detectors (`dlib`,
> `retina-face`) are commented out in `requirements.txt`; OpenCV's Haar cascade
> is the always-available fallback.

Verify the install with the offline smoke tests (no checkpoint download, runs on
CPU in seconds):

```bash
pytest tests/test_smoke.py -q
```

---

## Downloading datasets

Dataset download helpers live in `scripts/download/` (run from the repo root):

```bash
pip install -U huggingface_hub
python scripts/download/download_faceforensics.py   # -> data/FaceForensicsC23
python scripts/download/download_ntire_dataset.py    # -> data/NTIRE-RobustAIGenDetection-train
```

See `scripts/download/README.md` for details.

---

## Data preprocessing (FF++ videos → face crops)

Raw FF++ videos live under `data/FaceForensics++_C23/{real,fake/<manip>}/*.mp4`.
Convert them into training-ready 224×224 face crops with:

```bash
conda activate open_set_deepfake
python scripts/preprocess_ffpp.py                       # 4 classic manips + real
# options:
python scripts/preprocess_ffpp.py --limit-videos 2      # quick dry run
python scripts/preprocess_ffpp.py --manipulations Deepfakes Face2Face FaceSwap NeuralTextures FaceShifter
python scripts/preprocess_ffpp.py --frames-train 16 --image-size 256
```

The script samples evenly-spaced frames, detects faces with **MTCNN** (GPU),
takes the largest face, enlarges the box by 1.3× (squared), resizes to 224×224,
and writes `data/ffpp_frames/<split>/<manip>/<video>/<frame>.png`. Splits follow
the official FF++ protocol JSONs in `data/ffpp_splits/` (720/140/140 videos), so
identities never leak across train/val/test. The `DeepFakeDetection` subset uses
actor-based names outside the youtube-id split and is skipped by default.

## Data layout

The preprocessing script produces exactly the layout the datamodule expects
(per-split, per-manipulation, following the official FF++ protocol):

```
<root>/
  train/
    real/**/*.png                 # bona-fide      -> label 0, domain 0
    Deepfakes/**/*.png            # forgery domain 1
    Face2Face/**/*.png            # forgery domain 2
    FaceSwap/**/*.png             # forgery domain 3
    NeuralTextures/**/*.png       # forgery domain 4
  val/   (same structure)
  test/  (same structure)
```

Distinct manipulation subfolders become distinct **forgery source domains** for
FSM. Point the config at it with `data.root=/path/to/<root>`.

**Cross-manipulation** (train on 3 types, test on the held-out one): drop the
held-out manipulation folder from `train/` and `val/`, keep only it under a
separate test `root`. **Cross-dataset** (CDF, DFDC, WDF, FFIW, …): use a
manifest CSV instead:

```bash
python test.py ckpt_path=... data.source=manifest data.manifest=cdf.csv
```

where `cdf.csv` has columns `path,label,domain[,split]`.

Face cropping (dlib-style 1.3× margin → resize) is **off by default** (frames
assumed pre-cropped); enable with `data.use_face_crop=true data.face_backend=opencv`.

---

## Data preprocessing (NTIRE → manifest CSV)

The NTIRE 2026 Robust AI-Generated Image Detection dataset ships as 6 shards
(~277k images total, downloaded by `scripts/download/download_ntire_dataset.py`):

```
<root>/
  shard_0/
    images/<image_name>.jpg
    labels.csv            # (index), image_name, label (0 = real, 1 = fake)
  shard_1/ ... shard_5/
```

Build the `path,label,domain,split` manifest CSV that `configs/data/ntire.yaml`
expects (`source: manifest`) with:

```bash
conda activate open_set_deepfake
python scripts/preprocess_ntire.py \
  --data-root data/NTIRE-RobustAIGenDetection-train \
  --out data/ntire_manifest.csv
```

`--data-root` can point anywhere the shards already live, including a dataset
downloaded for another project on the same machine — the manifest stores
absolute image paths, so nothing needs to be copied into this repo. The split
is a stratified random split (by label) carved out of the labeled train pool,
since NTIRE's `-val` dataset has no trustworthy ground truth yet (see
`scripts/download/README.md`).

NTIRE has no per-generator domain info (`domain` == `label`, a single fake
domain), so **FSM has nothing to mix between domains and should be disabled**
when training on it:

```bash
./scripts/train.sh data=ntire fsm.enabled=false
```

---

## Training

```bash
conda activate open_set_deepfake
./scripts/train.sh data.root=/path/to/ffpp_frames

# common overrides
python train.py data.root=/data/ffpp optimizer.lr=3e-5 trainer.max_steps=30000
python train.py backbone=siglip2_large data.batch_size=16      # larger backbone
python train.py fsm.enabled=false peft.cdc.enabled=false        # ablations
python train.py trainer.strategy=ddp trainer.devices=4          # multi-GPU DDP
python train.py logger.wandb.offline=true                       # offline W&B
```

Defaults follow the paper (Sec. IV-A): Adam (`β=0.9/0.999`), `lr=3e-5`, no LR
decay, batch size 48, 30k steps, AUC as the validation metric, AMP on (`bf16-mixed`
by default; cuDNN benchmark on, `deterministic=warn`). Training supports DDP,
gradient accumulation/clipping, deterministic mode, resume (`ckpt_path=...`),
early stopping, checkpointing and LR scheduling — all via config. Both
**TensorBoard and W&B** log simultaneously (losses, AUC/ACC/F1/AP/EER, LR, GPU
memory, epoch time, ROC/PR/confusion figures, config); the best checkpoint is
uploaded as a W&B artifact.

For bit-exact reproducibility (slower), disable the performance-oriented
defaults: `trainer.precision=32-true deterministic=true trainer.benchmark=false`.

### Data augmentation (advanced, disabled by default)

The paper uses **no** augmentation during training, so it is **off by default**
(`data.augmentation.enabled=false`). An advanced, forgery-aware pipeline is
available and applied only to the training split when enabled — val/test always
use plain resize + normalise:

```bash
python train.py data.augmentation.enabled=true
# tune individual transforms:
python train.py data.augmentation.enabled=true \
  data.augmentation.jpeg=0.5 data.augmentation.gaussian_blur=0.3 \
  data.augmentation.random_erasing=0.25 data.augmentation.color_jitter=0.4
```

Available transforms (`configs/data/faceforensics.yaml → augmentation`):
horizontal flip, colour jitter, random resized crop, Gaussian blur, **random
JPEG recompression**, **random down/up-scaling** and random erasing (cutout).
The JPEG/downscale/blur transforms target compression and quality artefacts
relevant to face-forgery robustness. Each knob is a probability (or strength);
set it to `0` / `false` to disable that single transform.

---

## Evaluation

```bash
./scripts/eval.sh ckpt_path=outputs/DATE/TIME/checkpoints/last.ckpt
```

Reports ACC, AUC, F1, AP, EER, FPR, FNR, logs confusion/ROC/PR figures, and
writes `predictions.csv` (`path, prob, pred, label`).

---

## Inference

```bash
# single image
./scripts/infer.sh --ckpt model.ckpt --input face.png

# folder (+ CSV), with face cropping
./scripts/infer.sh --ckpt model.ckpt --input imgs/ --face-crop --csv preds.csv
```

Outputs prediction (`real`/`fake`), `P(fake)`, confidence and raw logit. FSM is
always disabled at inference.

---

## Configuration

Everything is configured through Hydra (`configs/`); no hyperparameters are
hardcoded. Key groups: `backbone`, `model`, `peft` (LoRA + CDC), `fsm`, `loss`,
`optimizer`, `scheduler`, `data`, `trainer`, `logger`, `callbacks`. Override any
value on the CLI (`group.key=value`) or swap a whole group (`backbone=siglip2_large`).

## Reproducing paper settings

| Setting | Value | Source |
|---|---|---|
| Optimizer / lr | Adam(0.9, 0.999), 3e-5, no decay | Sec. IV-A |
| Batch size / steps | 48 / 30k | Sec. IV-A |
| LoRA rank `r` | 8 | Sec. III-B |
| SCL margin / λ | 0.01 / 1.0 | Sec. III-D |
| FSM prob / δ prior | 0.5 / Beta(0.1, 0.1) | Sec. III-C |
| Real oversampling | ×4 | Sec. IV-B |
| Training data | FF++ c23, 4 manipulations | Sec. IV-A/B |

## Notes on faithfulness

- **CDC** uses the generalised Su et al. formulation `vanilla_conv − θ·x_c·Σw`;
  `θ=1` recovers the pure central-difference form of Eq. 3. Default `θ=0.7`
  (the well-tested CDCN value) is configurable via `peft.cdc.theta`.
- **SCL center** `C` is the per-batch mean of real features (Eq. 13), not a
  learned parameter.
- **FSM** pairs each fake sample with a fake sample from a *different* forgery
  domain, mixes AdaIN statistics (Eqs. 7–9), and is a no-op when fewer than two
  fake domains are present in a batch or in eval mode.

## License / citation

Cite the original OSDFD and SigLIP 2 papers (see `docs/`). This repository is a
re-implementation for research purposes.
