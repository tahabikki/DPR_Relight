#!/usr/bin/env python3
"""
Inference Script for Fine-tuned DPR Model (FIXED VERSION)

Clean shadow + directional-lighting removal for passport photos:
  1. Convert input RGB -> Lab, split into L, a, b
  2. Feed L through DPR with a FIXED flat-lighting SH
  3. Take the model's output L (relit luminance)
  4. Recombine with the ORIGINAL a, b channels  <-- keeps colors 100% faithful
  5. Convert Lab -> RGB, save

Because chroma (a, b) is never touched, the face identity, skin tone, hair color
and eye color are preserved exactly. Only the lighting/shadow pattern changes.

Usage:
    python scripts/infer.py \
        --checkpoint checkpoints/best_model.pth \
        --input path/to/photo.jpg \
        --output results/relit.jpg
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

# Project imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "model"))
sys.path.insert(0, str(project_root / "utils"))


# --------------------------------------------------------------------------- #
# Same flat SH vector as used in dataset.py during training                   #
# --------------------------------------------------------------------------- #
FLAT_PASSPORT_SH = np.array(
    [0.70, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    dtype=np.float32,
)


# --------------------------------------------------------------------------- #
# Device detection (same behavior as before)                                  #
# --------------------------------------------------------------------------- #
def detect_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.randn(1, device="mps")
            return torch.device("mps"), "🍎 Metal (Apple GPU)"
        except Exception:
            pass
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        return torch.device("cuda"), f"🎮 CUDA ({name}, {mem:.2f} GB)"
    return torch.device("cpu"), "💻 CPU"


# --------------------------------------------------------------------------- #
# Model loading                                                               #
# --------------------------------------------------------------------------- #
def load_model(checkpoint_path: str, model_variant: str, device: torch.device):
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"📦 Loading architecture for {model_variant}x{model_variant}...")
    if model_variant == "512":
        from defineHourglass_512_gray_skip import HourglassNet
        model = HourglassNet()
    elif model_variant == "1024":
        from defineHourglass_512_gray_skip import HourglassNet
        from defineHourglass_1024_gray_skip_matchFeature import HourglassNet_1024
        model = HourglassNet_1024(HourglassNet(), 16)
    else:
        raise ValueError(f"Unknown model variant: {model_variant}")

    print(f"📥 Loading checkpoint: {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model


# --------------------------------------------------------------------------- #
# Core inference: relight via L-channel replacement                           #
# --------------------------------------------------------------------------- #
def relight_passport(
    image_path: str,
    checkpoint_path: str,
    model_variant: str = "512",
    device: torch.device = None,
    model_size: int = 512,
) -> np.ndarray:
    """
    Returns a BGR uint8 image the same size as the input, with flat lighting.
    """
    if device is None:
        device, _ = detect_device()

    # -------- Load and prepare --------
    print(f"\n📷 Loading: {image_path}")
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    orig_h, orig_w = bgr.shape[:2]
    print(f"   Original size: {orig_w}x{orig_h}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # -------- Split to Lab and keep original a,b at ORIGINAL resolution --------
    # Important: we split at original resolution so a,b aren't resampled twice.
    lab_orig = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L_orig = lab_orig[:, :, 0]                  # uint8 [0..255], original size
    a_orig = lab_orig[:, :, 1]                  # uint8 [0..255], original size
    b_orig = lab_orig[:, :, 2]                  # uint8 [0..255], original size

    # -------- Resize L to the model's input size and normalize --------
    L_model = cv2.resize(L_orig, (model_size, model_size),
                         interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    print(f"   Model input L: {model_size}x{model_size}")

    # Tensors
    L_tensor = torch.from_numpy(L_model[None, None, ...]).float().to(device)   # [1,1,H,W]
    sh_tensor = torch.from_numpy(FLAT_PASSPORT_SH).float().reshape(1, 9, 1, 1).to(device)

    # -------- Load model and run --------
    print("🧠 Loading model...")
    model = load_model(checkpoint_path, model_variant, device)

    print("🚀 Running inference (flat lighting target)...")
    with torch.no_grad():
        out_L, _ = model(L_tensor, sh_tensor, 0)

    # -------- Convert model output back to uint8 L at original resolution -----
    out_L = torch.clamp(out_L, 0.0, 1.0)                          # safety
    out_L_np = out_L[0, 0].cpu().numpy()                          # [H, W]
    out_L_np = (out_L_np * 255.0).astype(np.uint8)
    out_L_resized = cv2.resize(out_L_np, (orig_w, orig_h),
                               interpolation=cv2.INTER_CUBIC)

    # -------- Recombine: new L + ORIGINAL a, b (chroma preserved) -------------
    lab_new = cv2.merge([out_L_resized, a_orig, b_orig])          # Lab
    rgb_new = cv2.cvtColor(lab_new, cv2.COLOR_LAB2RGB)
    bgr_new = cv2.cvtColor(rgb_new, cv2.COLOR_RGB2BGR)

    return bgr_new


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Run fine-tuned DPR for passport photo relighting.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model.pth or any trained checkpoint.")
    parser.add_argument("--input", required=True, help="Input photo path.")
    parser.add_argument("--output", required=True, help="Output path for the relit image.")
    parser.add_argument("--model-variant", default="512", choices=["512", "1024"])
    parser.add_argument("--model-size", type=int, default=512,
                        help="Model's spatial input size (512 or 1024).")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu", "mps"])
    args = parser.parse_args()

    print("=" * 70)
    print("DPR INFERENCE - PASSPORT PHOTO RELIGHTING (FIXED)")
    print("=" * 70)

    # Device
    if args.device == "auto":
        device, info = detect_device()
        print(f"🔍 Auto-detected device: {info}")
    else:
        device = torch.device(args.device if args.device != "mps" else "mps")
        print(f"📌 Manual device: {device}")

    try:
        result = relight_passport(
            image_path=args.input,
            checkpoint_path=args.checkpoint,
            model_variant=args.model_variant,
            device=device,
            model_size=args.model_size,
        )
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), result)
    print(f"\n✅ Output saved to: {out_path}")
    print("💡 Colors (a,b chroma) are preserved from input. Only lighting changed.")


if __name__ == "__main__":
    main()