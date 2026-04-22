#!/usr/bin/env python3
"""
Skin-Aware Inference for Passport Photo Relighting

Uses color-based skin detection to only process face/neck skin:
- Detects skin pixels using YCrCb color space
- Combines with face detection for precision
- Preserves: background, hair, eyes, mouth
- Only processes: actual skin (face/neck)
"""

import argparse
import sys
from pathlib import Path
import torch
import cv2
import numpy as np

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "model"))
sys.path.insert(0, str(project_root / "utils"))

from skin_mask import create_skin_mask

# Fixed flat passport SH lighting
FLAT_PASSPORT_SH = np.array([0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0], dtype=np.float32)


def detect_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.randn(1, device="mps")
            return torch.device("mps"), "Metal"
        except Exception:
            pass
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return torch.device("cuda"), f"CUDA ({name})"
    return torch.device("cpu"), "CPU"


def load_model(checkpoint_path: str, model_variant: str, device: torch.device):
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[Model] Loading architecture: {model_variant}x{model_variant}")
    if model_variant == "512":
        from defineHourglass_512_gray_skip import HourglassNet
        model = HourglassNet()
    else:
        from defineHourglass_512_gray_skip import HourglassNet
        from defineHourglass_1024_gray_skip_matchFeature import HourglassNet_1024
        model_512 = HourglassNet()
        model = HourglassNet_1024(model_512, 16)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def relight_skin_aware(
    image_path: str,
    checkpoint_path: str,
    model_variant: str = "512",
    device: torch.device = None,
    model_size: int = 512,
    blend_sigma: int = 5,
) -> np.ndarray:
    """
    Run skin-aware inference (only processes actual skin).
    """

    # Device
    if device is None:
        device, _ = detect_device()

    # Load image
    print(f"\n[Input] Loading: {image_path}")
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    orig_h, orig_w = bgr.shape[:2]
    print(f"  Size: {orig_w}x{orig_h}")

    # Create skin mask
    print("[Skin] Detecting skin regions...")
    skin_mask = create_skin_mask(bgr, with_neck=True)

    # Count skin pixels
    skin_pixels = np.count_nonzero(skin_mask)
    print(f"  Skin region: {skin_pixels} pixels ({100*skin_pixels/(orig_h*orig_w):.1f}%)")

    # Convert to RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Split to Lab at original resolution (keep chroma)
    lab_orig = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L_orig = lab_orig[:, :, 0]
    a_orig = lab_orig[:, :, 1]
    b_orig = lab_orig[:, :, 2]

    # Resize L for model
    L_model = cv2.resize(L_orig, (model_size, model_size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    print(f"  Model input: {model_size}x{model_size}")

    # Tensors
    L_tensor = torch.from_numpy(L_model[None, None, ...]).float().to(device)
    sh_tensor = torch.from_numpy(FLAT_PASSPORT_SH).float().reshape(1, 9, 1, 1).to(device)

    # Load model and run
    print("[Model] Loading...")
    model = load_model(checkpoint_path, model_variant, device)

    print("[Running] Inference...")
    with torch.no_grad():
        out_L, _ = model(L_tensor, sh_tensor, 0)

    # Get output L
    out_L = torch.clamp(out_L, 0.0, 1.0)
    out_L_np = out_L[0, 0].cpu().numpy()
    out_L_np = (out_L_np * 255.0).astype(np.uint8)
    out_L_resized = cv2.resize(out_L_np, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    # Recombine with original chroma (preserves skin color!)
    lab_new = cv2.merge([out_L_resized, a_orig, b_orig])
    rgb_new = cv2.cvtColor(lab_new, cv2.COLOR_LAB2RGB)
    bgr_new = cv2.cvtColor(rgb_new, cv2.COLOR_RGB2BGR)

    # Apply skin mask: smooth blend for no visible edges
    skin_mask_blur = cv2.GaussianBlur(skin_mask, (21, 21), 0)
    skin_mask_float = skin_mask_blur.astype(np.float32) / 255.0
    
    # Expand to 3 channels
    mask_3ch = np.stack([skin_mask_float] * 3, axis=-1)

    # Smooth blend: original BG + DPR on skin
    original_f = bgr.astype(np.float32) / 255.0
    processed_f = bgr_new.astype(np.float32) / 255.0

    blended = original_f * (1 - mask_3ch) + processed_f * mask_3ch
    result = np.clip(blended * 255, 0, 255).astype(np.uint8)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run skin-aware DPR inference")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    parser.add_argument("--input", required=True, help="Input photo")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--model-variant", default="512", choices=["512", "1024"])
    parser.add_argument("--model-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--blend", type=int, default=5, help="Blend sigma for smooth edges")
    args = parser.parse_args()

    print("=" * 60)
    print("SKIN-AWARE PASSPORT PHOTO RELIGHTING")
    print("=" * 60)

    # Device
    if args.device == "auto":
        device, info = detect_device()
        print(f"[Auto] Device: {info}")
    else:
        device = torch.device(args.device)
        print(f"[Manual] Device: {device}")

    try:
        result = relight_skin_aware(
            image_path=args.input,
            checkpoint_path=args.checkpoint,
            model_variant=args.model_variant,
            device=device,
            model_size=args.model_size,
            blend_sigma=args.blend,
        )
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), result)

    print(f"\n[OK] Saved: {out_path}")
    print("[INFO] Only skin processed. Hair/eyes/background preserved.")


if __name__ == "__main__":
    main()