# DPR Passport Photo Relighting

Fine-tune a deep learning model to automatically correct lighting and remove shadows from passport photos using deep Spherical Harmonics-based portrait relighting.

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
# Split your paired images into train/eval (80/20 split)
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
# Auto-detect GPU (CUDA, Metal, ROCm, or CPU fallback)
python scripts/train.py --config configs/finetune_passport.yaml

# Or specify device manually
python scripts/train.py --config configs/finetune_passport.yaml --device cuda
```

**Training output:**
- `checkpoints/best_model.pth` — best model checkpoint
- `checkpoints/training_curve.png` — loss curves

### 4. Inference on Your Images

```bash
# Run inference on images you want to check
python scripts/infer.py --checkpoint checkpoints/best_model.pth --input photo.jpg --output relit.jpg
```

---

## Technical Overview

**DPR (Deep Portrait Relighting)** — Hourglass CNN learns to relight portraits by:
1. Extracting low-frequency lighting (Spherical Harmonics coefficients)
2. Applying target lighting via learned decoder
3. Producing photorealistic relit output

**This project**: Fine-tunes DPR on real passport photo pairs with:
- ✅ Modern PyTorch 2.1 stack (Python 3.10+)
- ✅ Cross-platform GPU support (NVIDIA CUDA, Apple Metal, AMD ROCm)
- ✅ Frozen encoder transfer learning (prevents overfitting on small datasets)
- ✅ Centralized YAML config (no magic numbers)

---

## Dataset Format

Paired training images (input + target):

```
dataset/
├── input/          # Photos with poor lighting
│   ├── photo1.jpg
│   └── photo2.jpg
└── target/         # Same photos, well-lit
    ├── photo1.jpg
    └── photo2.jpg
```

```bash
python data/prepare_splits.py --src dataset --dst dataset_split --seed 42
```

This will:
- ✅ Verify all pairs match
---

## Configuration

Edit `configs/finetune_passport.yaml` to adjust:
- `model.freeze_encoder` — true for transfer learning (recommended for small datasets)
- `training.learning_rate` — 1e-4 for frozen encoder
- `training.num_epochs` — adjust based on convergence
- `training.batch_size` — reduce if CUDA OOM errors

---

## Project Structure

```
DPR Project/
├── README.md                       # This file
├── configs/finetune_passport.yaml  # Training hyperparameters
├── data/prepare_splits.py          # Dataset splitting script
├── dataset/                        # Your paired images (input/ + target/)
├── dataset_split/                  # Split data (created by prepare_splits.py)
├── model/                          # Hourglass architectures (512×512, 1024×1024)
├── scripts/
│   ├── train.py                    # Training script
│   ├── eval.py                     # Evaluation script
│   └── infer.py                    # Inference script
├── trained_model/                  # Pretrained checkpoints
├── checkpoints/                    # Training outputs (best_model.pth, curves, etc.)
└── archive/                        # Original project reference (not pushed)
```

---

## References

- **Original Paper**: [Deep Single Portrait Image Relighting](https://zhhoper.github.io/dpr.html) (Hao Zhou et al., ICCV 2019)
- **Original Code**: [zhhoper/DPR](https://github.com/zhhoper/DPR)

# Update GPU driver from nvidia.com
# Then verify:
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True

# If still not working, reinstall PyTorch with CUDA 13:
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

#### Apple Mac (M1/M2/M3/M4)
```bash
# Apple Metal should work automatically with PyTorch 1.12+
# If not, make sure you're using official Python, not Homebrew
python -c "import torch; print(torch.backends.mps.is_available())"
# Should print: True

# If False, reinstall PyTorch for Mac:
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio
```

#### AMD GPU (Windows/Linux)
```bash
# Verify ROCm installation:
pip list | grep torch
# Should show torch with rocm in version

# If missing, install PyTorch with ROCm:
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

**Verify GPU is working:**
```bash
python check_cuda.py
```

### CUDA Out of Memory (OOM)

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce `batch_size` in `configs/finetune_passport.yaml` (e.g., 8 → 4)
2. Use 512×512 model instead of 1024×1024
3. Enable CPU fallback: `python scripts/train.py --device cpu`
4. Reduce `image_size` (not recommended; affects quality)

### Dataset Not Found

**Error**: `FileNotFoundError: dataset_split not found`

**Solution**: Run dataset preparation first:
```bash
python data/prepare_splits.py --src dataset --dst dataset_split
```

### Pretrained Checkpoint Not Found

**Error**: `FileNotFoundError: trained_model/trained_model_03.t7 not found`

**Solution**: Ensure you have the original pretrained weights in `trained_model/`. Download from the original DPR repository if needed.

### SSIM Computation Error

**Error**: `ImportError: No module named 'skimage'`

**Solution**: Install scikit-image:
```bash
pip install scikit-image==0.22.0
```

---

## Questions & Support

For issues with:
- **Original DPR architecture**: See [zhhoper/DPR](https://github.com/zhhoper/DPR)
- **This fine-tuning implementation**: Check the troubleshooting section above or review code comments

---

**Last Updated**: 2026  
**Python Version**: 3.10+  
**PyTorch Version**: 2.1+
