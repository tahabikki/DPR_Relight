# DPR Passport Photo Relighting

**Automatically correct lighting and remove shadows from passport photos to meet ICAO compliance standards using deep learning.**

## Problem Statement

Passport photos must meet strict ICAO (International Civil Aviation Organization) lighting requirements:
- **Even illumination** across the face (no harsh shadows or highlights)
- **Neutral background** (white or off-white)
- **No red-eye or glare**
- **Proper exposure** without blown-out or underexposed regions

In real-world scenarios, photos taken with poor lighting equipment, harsh shadows, or non-uniform illumination frequently fail compliance checks and must be retaken, creating friction in government services and ID processing workflows.

**This project solves this** by automatically relighting passport photos to achieve neutral, compliant lighting while preserving the subject's face structure and features.

---

## Before / After

| Input (Poor Lighting) | Output (Corrected) |
|:---:|:---:|
| ![Placeholder](https://via.placeholder.com/200?text=Input+Photo) | ![Placeholder](https://via.placeholder.com/200?text=Relit+Photo) |

*Replace these images with actual before/after examples from your dataset.*

---

## Technical Approach

### DPR (Deep Single-Image Portrait Relighting)

This project fine-tunes the **DPR model** from Zhou et al. (ICCV 2019), which learns to relight portraits by manipulating **Spherical Harmonics (SH) lighting coefficients**.

#### How It Works

1. **Spherical Harmonics (SH)**: Light in 3D scenes can be compactly represented as a 9-dimensional vector of SH coefficients, capturing low-frequency environmental lighting.

2. **Hourglass Architecture**: DPR uses a stacked Hourglass CNN (similar to pose estimation networks) to:
   - **Encode** the input image into a compact feature representation
   - **Decode** the features with a target lighting vector to produce a relit output
   - **Jointly predict** shading and albedo for photorealistic results

3. **Spherical Harmonics Basis**: The model learns to apply lighting changes via SH coefficients:
   ```
   relit_image = relight(input_image, target_SH_lighting)
   ```

#### Key Innovation: Image-Pair Fine-tuning

The **original DPR** was trained on synthetic data with explicit SH lighting labels. This project adapts DPR for **real passport photo pairs** by:

- **Extracting SH coefficients on-the-fly** from the target (well-lit) image during training
- **Using diffuse estimation**: Assuming passport photos have relatively uniform face geometry and diffuse reflectance, we estimate SH from the overall brightness distribution
- **Pixel-space supervision**: Training loss compares the relit output directly with the target image, enabling end-to-end optimization

This approach is:
- ✅ **Minimally invasive** to the original architecture
- ✅ **Practical** for real paired data without manual SH annotation
- ✅ **Robust** to variations in albedo and geometry (passport photos are standardized)

---

## What's New vs. Original DPR

| Aspect | Original DPR | This Project |
|--------|--------------|--------------|
| **Input Data** | Synthetic renders + manual SH vectors | Real passport photo pairs (input + target) |
| **Training Target** | Arbitrary relighting with explicit SH | Neutral passport-compliant lighting |
| **SH Supervision** | Ground-truth SH coefficients | Estimated from target image brightness |
| **Loss Function** | Shading + SH prediction loss | Pixel-space L1 reconstruction loss |
| **Architecture** | Unchanged (Hourglass) | Unchanged (Hourglass) |
| **Dependencies** | PyTorch 0.4, Python 3.6 | PyTorch 2.1, Python 3.10+ |
| **Use Case** | Artistic relighting, research | Production passport photo correction |

---

## Dataset

### Format

The dataset consists of paired passport photos:

```
dataset/
├── input/          # Photos with poor/uneven lighting
│   ├── photo1_1.jpg
│   ├── photo1_2.jpg
│   └── ...
└── target/         # Same photos, professionally relit
    ├── photo1_1.jpg
    ├── photo1_2.jpg
    └── ...
```

**Pairing Rule**: Files with identical basenames across `input/` and `target/` folders are treated as pairs.

### Preparation & Splitting

Prepare your dataset into train/eval/test splits using the provided script:

```bash
python data/prepare_splits.py --src dataset --dst dataset_split --seed 42
```

This will:
- ✅ Verify all pairs match
- ✅ Split into **70% train / 20% eval / 10% test** (with fixed seed for reproducibility)
- ✅ Generate `.lst` files for DPR compatibility
- ✅ Report orphan files and statistics

**Output**:
```
dataset_split/
├── train/{input,target}/    # 708 pairs (70%)
├── eval/{input,target}/     # 202 pairs (20%)
├── test/{input,target}/     # 102 pairs (10%)
├── train.lst               # List file for training
├── eval.lst                # List file for evaluation
└── test.lst                # List file for testing
```

---

## Installation

### Prerequisites

- **Python**: 3.10 or 3.11
- **CUDA**: 11.8+ (optional, for GPU acceleration; CPU fallback supported)
- **GPU VRAM**: 
  - 512×512 model: ~4GB (can reduce batch size if needed)
  - 1024×1024 model: ~8GB

### Step 1: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv .venv
# On Windows:
.\.venv\Scripts\Activate.ps1
# On Linux/Mac:
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note on Modernization**: The original DPR used `PyTorch 0.4.1` and `Python 3.6`. This project upgrades to:
- `PyTorch 2.1.2` (2× speed improvement, better GPU support, cleaner API)
- `Python 3.10` (security, type hints, dict union operators)

All compatibility changes are minimal and non-breaking (see "Compatibility Changes" below).

### Step 2b: GPU Support (CUDA 13.0) - Optional but Recommended

If you have an NVIDIA GPU and want to accelerate training, install PyTorch with CUDA support:

**⚠️ Important**: If PyTorch is already installed, uninstall it first:

```bash
# Uninstall existing PyTorch (CPU-only version)
pip uninstall torch torchvision torchaudio -y

# Install PyTorch with CUDA 13.0 support (recommended)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Or for CUDA 11.8:
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Training Speed Comparison**:
- **GPU (NVIDIA)**: 1-2 hours for 50 epochs (batches processed in parallel)
- **CPU**: 10-20 hours (much slower, but always works)

### Step 3: Verify Installation

Verify both PyTorch and CUDA:

```bash
python check_cuda.py
```

Expected output with GPU:
```
PyTorch version: 2.1.2+cu130
CUDA available: True
CUDA device: NVIDIA GeForce RTX [YOUR_GPU_MODEL]
```

Expected output without GPU (CPU fallback):
```
PyTorch version: 2.1.2+cpu
CUDA available: False
CUDA: Not available - PyTorch CPU-only
```

The training script will **automatically detect and use GPU if available**, otherwise falls back to CPU.

---

## Usage

### 1. Prepare Dataset Splits

```bash
python data/prepare_splits.py --src dataset --dst dataset_split --seed 42
```

Expected output:
```
✓ Found 1012 paired images
✓ Split: Train 708 (70%) | Eval 202 (20%) | Test 102 (10%)
✓ No orphan files - all pairs matched!
✓ Created train.lst, eval.lst, test.lst
```

### 2. Fine-tune the Model

The training script **automatically detects and uses your GPU** (Apple Metal, NVIDIA CUDA, AMD ROCm) if available, otherwise falls back to CPU.

```bash
# Auto-detect GPU (recommended)
python scripts/train.py --config configs/finetune_passport.yaml

# Or manually specify device:
python scripts/train.py --config configs/finetune_passport.yaml --device cuda    # NVIDIA GPU
python scripts/train.py --config configs/finetune_passport.yaml --device mps     # Apple Metal (Mac/M-series)
python scripts/train.py --config configs/finetune_passport.yaml --device rocm    # AMD GPU
python scripts/train.py --config configs/finetune_passport.yaml --device cpu     # CPU only
```

**Device Autodetection Priority**:
1. 🍎 **Apple Metal** (Mac with M1/M2/M3/M4 chips) - fastest on Mac
2. 🎮 **NVIDIA CUDA** (GeForce RTX, A100, etc.) - fastest on Windows/Linux with NVIDIA
3. 🔴 **AMD ROCm** (Radeon RX, MI series) - fastest on Windows/Linux with AMD
4. 💻 **CPU** (fallback) - works everywhere but slower

**Expected Console Output**:
```
🔍 Auto-detecting device...
   🍎 Metal (Apple GPU)  # Or: 🎮 CUDA (NVIDIA GPU: RTX 3090, 24.00 GB)  # Or: 💻 CPU (No GPU detected)
```

**Performance** (approximate for 50 epochs, 708 training images):
- 🎮 **NVIDIA GPU (RTX 3090)**: ~1 hour
- 🍎 **Apple GPU (M3 Max)**: ~1.5 hours
- 🔴 **AMD GPU (RX 6900 XT)**: ~1.5 hours
- 💻 **CPU (Intel i7)**: ~10-20 hours

**Key Config Parameters** (in `configs/finetune_passport.yaml`):
- `model.variant`: "512" or "1024" (resolution)
- `model.freeze_encoder`: true/false (freeze backbone for transfer learning)
- `training.learning_rate`: 1e-4 (low for fine-tuning)
- `training.num_epochs`: 50 (adjust based on convergence)
- `training.batch_size`: 8 (reduce if CUDA OOM)

**Monitor Training**:
- Console output shows loss per epoch
- `checkpoints/training_curve.png` shows loss curves
- `checkpoints/best_model.pth` is saved when eval loss improves

Example output:
```
[Epoch 1/50] Train Loss: 0.042156 | Eval Loss: 0.038923
[Epoch 2/50] Train Loss: 0.038402 | Eval Loss: 0.036511
✅ Best checkpoint saved: checkpoints/best_model.pth
...
```

### 3. Evaluate on Test Set

```bash
python scripts/eval.py \
  --checkpoint checkpoints/best_model.pth \
  --eval-split test \
  --model-variant 512
```

This computes:
- **PSNR** (Peak Signal-to-Noise Ratio): Higher is better, typical 22-30 dB
- **SSIM** (Structural Similarity): 0-1 scale, higher is better
- **LPIPS** (Learned Perceptual Similarity): Lower is better

Saves visual comparison grid to `checkpoints/eval_samples_test.png`.

### 4. Run Inference on New Images

```bash
python scripts/infer.py \
  --checkpoint checkpoints/best_model.pth \
  --input my_passport_photo.jpg \
  --output relit_photo.jpg \
  --model-variant 512
```

For batch processing:
```bash
for img in input_photos/*.jpg; do
  python scripts/infer.py \
    --checkpoint checkpoints/best_model.pth \
    --input "$img" \
    --output "output_photos/$(basename $img)" \
    --model-variant 512
done
```

---

## Results

### Quantitative Metrics (Placeholder — Fill After Training)

| Metric | Train | Eval | Test |
|--------|-------|------|------|
| **PSNR (dB)** | [fill after training] | [fill after training] | [fill after training] |
| **SSIM** | [fill after training] | [fill after training] | [fill after training] |
| **LPIPS** | [fill after training] | [fill after training] | [fill after training] |

### Qualitative Results

Visual comparison grid (input | prediction | target) is saved during evaluation:
- `checkpoints/eval_samples_eval.png` (20 random eval samples)
- `checkpoints/eval_samples_test.png` (20 random test samples)

### Training Dynamics

The `checkpoints/training_curve.png` plot shows:
- Train loss steadily decreasing (model learning)
- Eval loss converging (generalization)
- Gap between train/eval indicating regularization effectiveness

---

## Project Structure

```
.
├── README.md                      # This file
├── requirements.txt               # Python dependencies (pinned versions)
├── .gitignore                     # Git ignore rules
│
├── configs/
│   └── finetune_passport.yaml    # Hyperparameters (no magic numbers!)
│
├── data/
│   ├── prepare_splits.py         # Dataset splitting script
│   ├── dataset.py                # PyTorch Dataset + DataLoader
│   ├── example_light/            # Reference SH lighting vectors (for reference)
│   ├── obama.jpg                 # Demo image (from original DPR)
│   ├── train.lst                 # Original DPR list files (kept for reference)
│   ├── val.lst
│   └── test.lst
│
├── dataset/                       # Your paired dataset (input/ + target/)
│   ├── input/
│   └── target/
│
├── dataset_split/                 # Split dataset (created by prepare_splits.py)
│   ├── train/{input,target}/
│   ├── eval/{input,target}/
│   ├── test/{input,target}/
│   ├── train.lst
│   ├── eval.lst
│   └── test.lst
│
├── model/
│   ├── defineHourglass_512_gray_skip.py        # Hourglass 512×512 architecture
│   └── defineHourglass_1024_gray_skip_matchFeature.py  # Hourglass 1024×1024 variant
│
├── utils/
│   ├── utils_SH.py               # Spherical Harmonics utilities
│   ├── utils_normal.py           # Normal vector handling
│   └── utils_shtools.py          # Advanced SH tools (optional)
│
├── scripts/
│   ├── train.py                  # Fine-tuning entry point
│   ├── eval.py                   # Evaluation on test/eval split
│   └── infer.py                  # Single-image inference
│
├── checkpoints/                   # Training outputs (git-ignored)
│   ├── best_model.pth            # Best checkpoint (lowest eval loss)
│   ├── final_model.pth           # Final model after all epochs
│   ├── training_curve.png        # Loss curves
│   ├── eval_samples_eval.png     # Visual comparison on eval set
│   └── eval_samples_test.png     # Visual comparison on test set
│
├── trained_model/                 # Pretrained DPR checkpoints (for fine-tuning)
│   ├── trained_model_03.t7       # 512×512 pretrained
│   └── trained_model_1024_03.t7  # 1024×1024 pretrained
│
└── archive/                       # Original project artifacts (for reference)
    ├── README.md                 # Archive manifest
    ├── README_Finetune.md        # Original requirements document
    ├── demos/
    │   ├── testNetwork_demo_512.py
    │   └── testNetwork_demo_1024.py
    └── result/                   # Original demo outputs
```

---

## Limitations & Future Work

### Known Limitations

1. **Dataset Bias**: Currently trained on your dataset. Results may not generalize perfectly to:
   - Non-frontal faces (passport photos are frontal by definition, but variations exist)
   - Significantly different ethnicities if training data is homogeneous
   - Extreme lighting (very dark or backlit scenes)
   - Faces with heavy makeup or scars (not represented in training)

2. **Fixed Output Lighting**: The model is trained to produce a single "neutral" passport-compliant lighting. It does not support arbitrary relighting like original DPR.

3. **Face-Only**: Only processes frontal face regions. Background and body are minimally adjusted.

4. **Dependency on Face Detection**: Currently assumes input is already roughly centered on the face. Future versions could add face detection + alignment.

### Future Improvements

- [ ] **Multi-scale training**: Train on both 512×512 and 1024×1024 simultaneously
- [ ] **Confidence estimation**: Output per-pixel confidence scores for compliance verification
- [ ] **Lightweight inference**: Distill to a smaller model for edge deployment
- [ ] **Iterative refinement**: Apply model multiple times for harder cases
- [ ] **Face parsing**: Segment face regions and apply targeted relighting
- [ ] **Real-time processing**: Optimize for video stream processing
- [ ] **Generative augmentation**: Use diffusion models to synthesize diverse lighting variations
- [ ] **ICAO validator**: Integrate automated ICAO compliance checker in pipeline

---

## Compatibility Changes from Original DPR

The original DPR codebase has been minimally modified for modern Python and PyTorch. Here are all breaking changes:

| File | Change | Reason |
|------|--------|--------|
| Model loading | `.t7` → `.pth` format in some paths | PyTorch 2.x prefers `.pth`; `.t7` still supported |
| Tensor creation | `torch.cuda()` → `tensor.to(device)` | Modern PyTorch best practice |
| Printing | `print()` instead of `sys.stdout.write()` | Python 3 idiom |
| Path handling | `pathlib.Path` instead of string concat | Cross-platform (Windows/Linux) compatibility |
| None other | ✅ Hourglass architecture unchanged | Ensures compatibility with pretrained weights |

**Backward Compatibility**: All changes are backward-compatible. Existing `.t7` checkpoints load without modification.

---

## Credits

### Original Paper & Code

- **Paper**: [Deep Single Portrait Image Relighting](https://zhhoper.github.io/dpr.html)  
  Hao Zhou, Sunil Hadap, Kalyan Sunkavalli, David W. Jacobs  
  *ICCV 2019*

- **Original Repository**: [zhhoper/DPR](https://github.com/zhhoper/DPR)  
  Implements the Hourglass architecture and SH lighting framework

### This Fine-tuning Project

- **Adaptation**: Fine-tuning for passport photo relighting
- **Dataset Preparation**: Custom paired dataset splitting
- **Training Pipeline**: Modern PyTorch training loop with checkpointing
- **Evaluation**: PSNR, SSIM, LPIPS metrics + visual comparisons

---

## License

This project inherits the license from the original DPR repository. Please check the original DPR repository for licensing terms. Typically research code from academic papers is released under:
- **CC-BY-NC-SA 4.0** (Non-commercial, attribution required)
- or **MIT/Apache 2.0** for open-source variants

When deploying this model in production or commercial settings, verify licensing compliance with the original authors.

---

## Citation

If you use this fine-tuning code or dataset in research, please cite the original DPR paper:

```bibtex
@InProceedings{DPR,
  title={Deep Single Portrait Image Relighting},
  author={Hao Zhou and Sunil Hadap and Kalyan Sunkavalli and David W. Jacobs},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2019}
}
```

And optionally credit this fine-tuning adaptation in your methods section.

---

## Troubleshooting

### GPU Not Detected (Always Using CPU)

**Symptom**: Console shows `💻 Device: CPU` even though you have a GPU

**Solutions** (by GPU type):

#### NVIDIA GPU (Windows/Linux)
```bash
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
