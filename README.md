# DPR Passport Photo Relighting

Fine-tune a deep learning model to automatically correct lighting and remove shadows from passport photos.

---

## Quick Start

### 1. Setup

```bash
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Organize paired images:
# dataset/
#   ├── input/     (photos with poor lighting)
#   └── target/    (same photos, well-lit)

# Split into train/eval
python data/prepare_splits.py --src dataset --dst dataset_split --seed 42
```

### 3. Train

```bash
python scripts/train.py --config configs/finetune_passport.yaml
```

### 4. Inference

```bash
# Option 1: Skin-aware (RECOMMENDED)
python scripts/infer_skin.py --checkpoint checkpoints/best_model.pth --input photo.jpg --output relit.jpg

# Option 2: Face-aware
python scripts/infer_face.py --checkpoint checkpoints/best_model.pth --input photo.jpg --output relit.jpg

# Option 3: Whole image
python scripts/infer.py --checkpoint checkpoints/best_model.pth --input photo.jpg --output relit.jpg
```

---

## Inference Options

| Script | What it processes | Best for |
|--------|----------------|---------|
| `infer_skin.py` | Only skin pixels (44%) | **Best** - preserves hair, eyes, background |
| `infer_face.py` | Face rectangle + neck | Simple, fast |
| `infer.py` | Whole image | Quick testing |

**Recommended**: Use `infer_skin.py` - it detects skin using color (YCrCb + HSV + RGB) and only processes actual skin, preserving:
- Background ✓
- Hair ✓
- Eyes ✓
- Mouth ✓

---

## How It Works

1. **Training**: Model learns L→L mapping with fixed flat SH
2. **Inference**: 
   - Detect skin regions (color-based)
   - Run DPR only on skin
   - Blend with original (preserves non-skin areas)

---

## Configuration

Edit `configs/finetune_passport.yaml`:

| Parameter | Default | Description |
|-----------|---------|------------|
| `num_epochs` | 100 | Training epochs |
| `batch_size` | 8 | Batch size |
| `learning_rate` | 1e-4 | Learning rate |
| `freeze_encoder` | true | Freeze backbone |

---

## Project Structure

```
DPR Project/
├── README.md
├── configs/finetune_passport.yaml
├── requirements.txt
├── data/
│   ├── prepare_splits.py
│   └── dataset.py
├── utils/
│   ├── skin_mask.py      # Color-based skin detection
│   └── face_mask.py   # Face detection
├── scripts/
│   ├── train.py
│   ├── infer.py        # Whole image
│   ├── infer_face.py  # Face detection
│   └── infer_skin.py # Skin detection (BEST)
├── model/
├── dataset/
├── dataset_split/
├── trained_model/
└── checkpoints/
```

---

## Troubleshooting

### CUDA out of memory
```yaml
training:
  batch_size: 4  # Reduce
```

### Face not detected
Use `infer_skin.py` - works without face detection

### Skin detection too aggressive/conservative
Edit `utils/skin_mask.py` - adjust color ranges

---

## Reference

- **Paper**: [Deep Single Portrait Image Relighting](https://zhhoper.github.io/dpr.html)
- **Code**: [zhhoper/DPR](https://github.com/zhhoper/DPR)

---

**Last Updated**: 2026
**Python**: 3.10+
**PyTorch**: 2.1+