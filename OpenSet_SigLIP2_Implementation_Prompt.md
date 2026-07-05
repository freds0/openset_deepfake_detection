# Open-Set Deepfake Detection with SigLIP 2

## Role

Act as a **Senior Computer Vision Engineer**, **Machine Learning
Research Engineer**, and **PyTorch Lightning Architect**.

Your goal is to build a **production-ready, research-oriented
repository** for **Open-Set Deepfake Detection** by re-implementing the
method proposed in:

> **Open-Set Deepfake Detection: A Parameter-Efficient Adaptation Method
> with Forgery Style Mixture**

The implementation must replace the original ImageNet/ViT backbone with
**SigLIP 2** while preserving the methodology proposed in the paper.

------------------------------------------------------------------------

# 1. Mandatory Reading

Before writing any code, thoroughly read every paper available in the
local `docs/` directory.

Do not rely on prior knowledge or paper abstracts.

The implementation must faithfully reproduce the methodology described
in the papers.

## Study in detail

### Open-Set Deepfake Detection

Understand:

-   overall architecture
-   Forgery Style Mixture (FSM)
-   CDC Adapter
-   LoRA-based PEFT
-   Single-Center Loss
-   training pipeline
-   inference pipeline
-   hyperparameters
-   ablation studies
-   implementation details

### SigLIP 2

Understand:

-   visual encoder architecture
-   MAP pooling
-   patch token representations
-   feature extraction
-   positional embeddings
-   preprocessing
-   supported resolutions
-   differences from CLIP
-   recommended frozen-backbone usage

Also inspect any additional papers related to:

-   LoRA
-   PEFT
-   Adapter tuning
-   Vision Transformers
-   Open-set recognition
-   Deepfake detection

When implementation details are ambiguous, always prioritize the papers.

------------------------------------------------------------------------

# 2. Repository Inspection

Before implementing anything:

-   inspect the complete repository
-   understand existing modules
-   reuse existing utilities
-   avoid duplicated functionality
-   keep the coding style consistent

Before implementing each module:

1.  Read the corresponding paper section.
2.  Check whether similar functionality already exists.
3.  Extend existing code whenever possible.
4.  Add comments referencing the corresponding paper section or figure.

------------------------------------------------------------------------

# 3. Model Architecture

## Backbone

Use **SigLIP 2** as the visual encoder.

Default checkpoint:

``` text
google/siglip2-base-patch16-224
```

Requirements:

-   frozen backbone by default
-   extract MAP pooled embedding
-   extract last-layer patch tokens
-   configurable training modes:
    -   frozen
    -   LoRA
    -   LoRA + CDC Adapter
    -   full fine-tuning (ablation only)

## PEFT

Implement:

### LoRA

-   attention Q/K/V
-   configurable rank
-   configurable alpha
-   configurable dropout

### CDC Adapter

Insert CDC adapters into transformer FFN blocks.

Requirements:

-   Conv1x1 down projection
-   Central Difference Convolution
-   Conv1x1 up projection
-   residual connection
-   operates on reshaped patch tokens

## Forgery Style Mixture

Implement FSM exactly as described in the paper.

Requirements:

-   active only during training
-   applied only to fake samples
-   feature statistic mixing using AdaIN-style statistics
-   configurable probability
-   disabled during validation and inference

## Classification Head

Implement a configurable MLP classifier using:

-   pooled embedding
-   optional global + local feature fusion

## Loss

Use:

-   BCEWithLogitsLoss
-   Single-Center Loss

Total loss:

L = BCE + λ × SCL

------------------------------------------------------------------------

# 4. Data Pipeline

Implement a LightningDataModule supporting FaceForensics++ style
datasets.

Features:

-   image folders
-   extracted video frames
-   manipulation labels
-   binary labels
-   optional face detection
-   RetinaFace/OpenCV support
-   SigLIP2 preprocessing

------------------------------------------------------------------------

# 5. Training Framework

The project **must be implemented entirely with PyTorch Lightning 2.x**.

Do not implement custom training loops.

Implement:

-   LightningModule
-   LightningDataModule
-   configure_optimizers()
-   training_step()
-   validation_step()
-   test_step()
-   predict_step()

Support:

-   AMP
-   DDP
-   gradient accumulation
-   gradient clipping
-   resume training
-   deterministic training
-   early stopping
-   checkpointing
-   LR schedulers

------------------------------------------------------------------------

# 6. Logging

Support **both WandB and TensorBoard simultaneously**.

Use:

-   WandbLogger
-   TensorBoardLogger

Automatically log:

-   losses
-   AUC
-   Accuracy
-   F1
-   AP
-   EER
-   learning rate
-   GPU memory
-   training time
-   ROC curves
-   Precision-Recall curves
-   confusion matrices
-   hyperparameters
-   configuration files

Upload the best checkpoint as a WandB artifact.

Support offline WandB mode.

------------------------------------------------------------------------

# 7. Evaluation

Provide scripts for:

-   in-domain evaluation
-   cross-manipulation evaluation
-   cross-dataset evaluation

Report:

-   Accuracy
-   AUC
-   F1
-   AP
-   EER
-   FPR
-   FNR
-   confusion matrix
-   CSV predictions

------------------------------------------------------------------------

# 8. Inference

Create inference scripts for:

-   single image
-   folder of images

Output:

-   prediction
-   probability
-   confidence
-   raw logits

FSM must be disabled during inference.

------------------------------------------------------------------------

# 9. Project Structure

Use a modular architecture:

``` text
configs/
docs/
scripts/
src/
    data/
    models/
    losses/
    lightning/
    training/
    inference/
    utils/
train.py
test.py
predict.py
README.md
requirements.txt
```

------------------------------------------------------------------------

# 10. Configuration

Use YAML configuration files for everything.

No hyperparameters may be hardcoded.

Include:

-   backbone
-   LoRA
-   CDC
-   FSM
-   optimizer
-   scheduler
-   trainer
-   callbacks
-   loggers
-   dataset
-   checkpoints
-   mixed precision
-   distributed training

Hydra or OmegaConf should be used.

------------------------------------------------------------------------

# 11. Engineering Requirements

The repository must follow production-quality software engineering
practices.

Requirements:

-   PyTorch Lightning 2.x
-   Hydra/OmegaConf
-   type hints
-   docstrings
-   modular architecture
-   reproducible experiments
-   unit-test friendly design

Only the following parameters should be trainable by default:

-   LoRA
-   CDC Adapter
-   classification head
-   optional normalization layers

------------------------------------------------------------------------

# 12. Deliverables

Provide:

-   complete source code
-   YAML configurations
-   training scripts
-   evaluation scripts
-   inference scripts
-   README
-   installation guide
-   reproducible experiments
-   comments linking implementation to the corresponding paper sections

------------------------------------------------------------------------

## Development Environment

All development, training, evaluation, and inference must be performed
using the Conda environment named:

``` bash
conda activate open_set_deepfake
```

Requirements:

-   Assume this environment already exists.
-   Do not create a new Conda environment or use a different environment
    name.
-   Ensure all installation instructions, scripts, and examples use
    `open_set_deepfake`.
-   When adding new dependencies, install them into this environment and
    update `requirements.txt` accordingly.
-   All shell scripts (`train.sh`, `eval.sh`, `infer.sh`) and the README
    should explicitly activate this environment before running any
    command.
