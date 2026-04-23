# DPR Passport Photo Relighting

Fine-tune DPR model to automatically correct directional lighting in passport photos - normalize bright and dark sides, remove harsh shadows.

---

## Quick Start

### 1. Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# dataset/
#   ├── input/     (photos with harsh lighting)
#   └── target/    (same photos, well-lit reference)

python data/prepare_splits.py --src dataset --dst dataset_split --seed 42
```

### 3. Train

```bash
python scripts/train.py --config configs/finetune_passport.yaml
```

### 4. Inference (Recommended)

```bash
python scripts/infer_skin.py --checkpoint checkpoints/best_model.pth --input photo.jpg --output relit.jpg
```

---

## What This Does

**Problem**: Passport photos with directional window light have:
- Bright side: overexposed, near-white skin
- Dark side: harsh shadow, too dark

**Solution**: Three-step processing:
1. **Highlight compression** - prevents overexposed areas from clipping
2. **DPR relighting** - normalizes to flat ambient lighting
3. **Chroma correction** - matches skin tone across face

**Result**: Even, passport-compliant lighting on both sides of the face.

---

## Inference Scripts

| Script | Use Case |
|--------|----------|
| `infer_skin.py` | **BEST** - Only skin pixels, feathered blend |
| `infer_face.py` | Face detection, ellipse mask |
| `infer.py` | Whole image, no mask |

**Recommended**: `infer_skin.py` - only processes actual skin, preserves everything else.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Skin detection** | Color-based (YCrCb + HSV + RGB), follows face shape |
| **Highlight compression** | Soft-knee rolloff (knee=0.70, ceiling=0.85) |
| **Flat ambient SH** | [0.5, 0, 0, 0, 0, 0, 0, 0, 0] - truly neutral |
| **Chroma correction** | Pulls lit-side toward neutral skin tone |
| **Feathered blend** | ~41px Gaussian blur for smooth edges |

---

## Configuration

Edit `configs/finetune_passport.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | 50 | Training epochs (100 max) |
| `batch_size` | 4 | Batch size (reduce if OOM) |
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
│   ├── prepare_splits.py       # Dataset splitting
│   └── dataset.py              # Training data loader (with skin mask)
├── utils/
│   ├── skin_mask.py           # Color-based skin detection
│   └── face_mask.py           # Face detection (ellipse mask)
├── scripts/
│   ├── train.py               # Training
│   ├── infer.py               # Whole image inference
│   ├── infer_face.py          # Face detection inference
│   └── infer_skin.py           # Skin-aware inference (BEST)
├── model/
│   └── defineHourglass_512_gray_skip.py
├── dataset/
│   └── dataset_split/
├── trained_model/
└── checkpoints/
```

---

## How It Works

### Training
1. Load paired images (bad lighting → good lighting)
2. Detect skin, apply mask
3. Extract L channel, apply skin mask
4. Train DPR to map L(bad) → L(good) with flat SH

### Inference
1. Load image, detect skin mask
2. **Compress highlights** (knee=0.70, ceiling=0.85)
3. Feed L to DPR with flat ambient SH
4. **Correct chroma** on lit-side pixels
5. **Feathered blend** - only skin changes, rest preserved

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce `batch_size: 4` in config |
| Visible rectangle in output | Use `infer_skin.py` |
| Bright side still overexposed | Normalized by highlight compression |
| Color mismatch on lit side | Corrected by chroma correction |

---

## Reference

- **Paper**: [Deep Single Portrait Image Relighting](https://zhhoper.github.io/dpr.html)
- **Code**: [zhhoper/DPR](https://github.com/zhhoper/DPR)

---

**Last Updated**: 2026
**Python**: 3.10+
**PyTorch**: 2.1+