# Dataset download scripts

Helpers that fetch datasets from the HuggingFace Hub. **Run them from the
repository root** (paths are relative to the current working directory) inside
the project environment:

```bash
conda activate open_set_deepfake
pip install -U huggingface_hub
python scripts/download/download_faceforensics.py   # -> data/FaceForensics++_C23
python scripts/download/download_ntire_dataset.py    # -> data/NTIRE-RobustAIGenDetection-train
python scripts/download/download_ntire_val_dataset.py # -> data/NTIRE-RobustAIGenDetection-val
```

| Script | Dataset | Output |
|---|---|---|
| `download_faceforensics.py` | `bitmind/FaceForensicsC23` | `data/FaceForensics++_C23` |
| `download_ntire_dataset.py` | `deepfakesMSU/NTIRE-RobustAIGenDetection-train` | `data/NTIRE-RobustAIGenDetection-train` |
| `download_ntire_val_dataset.py` | `deepfakesMSU/NTIRE-RobustAIGenDetection-val` | `data/NTIRE-RobustAIGenDetection-val` |

The NTIRE test split is not released yet (TBD by the challenge organizers).

After downloading FaceForensics++ videos, convert them into training-ready face
crops with `scripts/preprocess_ffpp.py` (see the main README).
