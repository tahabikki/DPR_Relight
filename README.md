# DPR Passport Photo Relighting

Fine-tune a deep learning model to automatically correct lighting and remove shadows from passport photos using Spherical Harmonics-based portrait relighting.

---

## Quick Start

### 1. Setup

```bash
# Create virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Organize your paired images first:
# dataset/
#   ├── input/     (photos with poor lighting)
#   └── target/    (same photos, well-lit - reference)

# Split into train/eval (80/20)
python data/prepare_splits.py --src dataset --dst dataset_split --seed 42
```

Expected structure:
```
dataset_split/
├── train/{input,target}/    # 80% of images
├── eval/{input,target}/     # 20% of images  
└── train.lst, eval.lst
```

### 3. Train

```bash
# Auto-detect GPU
python scripts/train.py --config configs/finetune_passport.yaml

# Or specify device manually
python scripts/train.py --config configs/finetune_passport.yaml --device cuda
```

**Training output:**
- `checkpoints/best_model.pth` — best model checkpoint
- `checkpoints/training_curve.png` — loss curves

### 4. Inference

```bash
# Relight any passport photo
python scripts/infer.py --checkpoint checkpoints/best_model.pth --input photo.jpg --output relit.jpg
```

---

## How It Works

1. **Training**: Model learns to map Luminance (L) channel from bad-lit photos to flat passport lighting using fixed SH coefficients
2. **Inference**: Input is split into L,a,b channels → L is relit → recombined with original a,b (colors preserved)

**Key improvements:**
- Uses L channel only (not RGB) to match pretrained model architecture
- Fixed flat passport SH lighting (no fake estimation)
- Chroma (a,b) preserved from input → natural colors

---

## Project Structure

```
DPR Project/
├── README.md                       # This file
├── configs/finetune_passport.yaml  # Training config
├── data/
│   ├── prepare_splits.py          # Split dataset
│   └── dataset.py                # Data loader
├── dataset/                     # Your paired images
├── dataset_split/               # Split data
├── model/                      # Hourglass architectures
├── scripts/
│   ├── train.py                # Training
│   ├── infer.py               # Inference
│   └── eval.py                # Evaluation
├── trained_model/              # Pretrained weights
├── checkpoints/                # Your trained models
└── requirements.txt
```

---

## Configuration

Edit `configs/finetune_passport.yaml`:

| Parameter | Description | Default |
|-----------|------------|---------|
| `model.freeze_encoder` | Freeze backbone for transfer learning | true |
| `training.learning_rate` | Learning rate | 1e-4 |
| `training.num_epochs` | Training epochs | 100 |
| `training.batch_size` | Batch size | 8 |

---

## Troubleshooting

### Training produces artifacts/blurry output

**Cause**: Using wrong checkpoint or old data format

**Solution**:
```bash
# Delete old checkpoints
rmdir /S /Q checkpoints

# Retrain fresh
python scripts/train.py --config configs/finetune_passport.yaml
```

### Inference colors look wrong

**Cause**: Old infer.py version

**Solution**: Ensure you're using the latest version with LAB recombination (check for `cv2.COLOR_LAB2RGB` in infer.py)

### CUDA out of memory

**Solution**: Decrease batch size in config:
```yaml
training:
  batch_size: 4
```

---

## Reference

- **Original Paper**: [Deep Single Portrait Image Relighting](https://zhhoper.github.io/dpr.html) (ICCV 2019)
- **Original Code**: [zhhoper/DPR](https://github.com/zhhoper/DPR)

---

**Last Updated**: 2026
**Python**: 3.10+
**PyTorch**: 2.1+
