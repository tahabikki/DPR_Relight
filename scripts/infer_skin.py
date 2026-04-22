#!/usr/bin/env python3
"""
Skin-Aware Inference - NORMALIZE DIRECTIONAL LIGHT

Fixes:
1. Proper flat SH (truly ambient, no directional)
2. Highlight compression before DPR (gives headroom to pull down bright side)
3. Chromatic correction on lit side (warm cast removal)
4. Feathered mask for smooth blend
"""

import argparse
import sys
from pathlib import Path
import torch
import cv2
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "model"))
sys.path.insert(0, str(project_root / "utils"))

from skin_mask import create_skin_mask

# FIX 1: Truly flat ambient SH - only DC term, everything else exactly zero
# 0.5 = neutral brightness (0.7 was too bright)
FLAT_PASSPORT_SH = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def detect_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "CUDA"
    return torch.device("cpu"), "CPU"


def load_model(checkpoint_path, model_variant, device):
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


def compress_highlights(L, knee=0.70, ceiling=0.85):
    """Soft-knee highlight compression."""
    out = L.copy()
    above = L > knee
    if not np.any(above):
        return out
    t = (L[above] - knee) / (1.0 - knee)
    t = 1.0 - (1.0 - t) ** 2  # ease-out
    out[above] = knee + t * (ceiling - knee)
    return out


def correct_lit_side_chroma(L, a, b, lit_start=0.65, lit_full=0.90, strength=0.6):
    """Pull lit-side chroma toward midtone skin."""
    skin_mid = (L > 0.35) & (L < lit_start)
    if skin_mid.sum() < 100:
        return a, b
    
    a_neutral = float(np.median(a[skin_mid]))
    b_neutral = float(np.median(b[skin_mid]))
    
    lit_w = np.clip((L - lit_start) / (lit_full - lit_start), 0, 1).astype(np.float32)
    lit_w *= strength
    
    a_new = a.astype(np.float32) * (1 - lit_w) + a_neutral * lit_w
    b_new = b.astype(np.float32) * (1 - lit_w) + b_neutral * lit_w
    
    return np.clip(a_new, 0, 255).astype(np.uint8), np.clip(b_new, 0, 255).astype(np.uint8)


def relight_normalized(image_path, checkpoint_path, model_variant="512",
                     device=None, model_size=512):
    if device is None:
        device, _ = detect_device()

    print(f"[Input] {image_path}")
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    
    orig_h, orig_w = bgr.shape[:2]
    print(f"  Size: {orig_w}x{orig_h}")

    print("[Mask] Detecting skin...")
    skin_mask = create_skin_mask(bgr, with_neck=True)
    skin_pixels = np.count_nonzero(skin_mask)
    print(f"  Skin: {skin_pixels} pixels ({100*skin_pixels/(orig_h*orig_w):.1f}%)")

    # Convert to Lab
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32) / 255.0
    a_orig = lab[:, :, 1]
    b_orig = lab[:, :, 2]

    # FIX 2: Compress highlights before DPR
    print("[Preprocess] Compressing highlights (knee=0.70, ceiling=0.85)...")
    L_compressed = compress_highlights(L, knee=0.70, ceiling=0.85)

    # Feed FULL compressed L to DPR
    L_model = cv2.resize(L_compressed, (model_size, model_size), interpolation=cv2.INTER_AREA)

    L_tensor = torch.from_numpy(L_model[None, None, ...]).float().to(device)
    sh_tensor = torch.from_numpy(FLAT_PASSPORT_SH).float().reshape(1, 9, 1, 1).to(device)

    print("[Model] Loading...")
    model = load_model(checkpoint_path, model_variant, device)

    print("[Running] DPR with flat ambient SH...")
    with torch.no_grad():
        out_L, _ = model(L_tensor, sh_tensor, 0)

    out_L = torch.clamp(out_L, 0.0, 1.0)
    out_L_np = (out_L[0, 0].cpu().numpy() * 255).astype(np.uint8)
    out_L_resized = cv2.resize(out_L_np, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    # FIX 3: Correct chroma on lit side
    print("[Postprocess] Correcting lit-side chroma...")
    a_corrected, b_corrected = correct_lit_side_chroma(
        L, a_orig, b_orig, lit_start=0.65, lit_full=0.90, strength=0.6
    )

    # Recombine
    lab_new = cv2.merge([out_L_resized, a_corrected, b_corrected])
    rgb_new = cv2.cvtColor(lab_new, cv2.COLOR_LAB2RGB)
    bgr_new = cv2.cvtColor(rgb_new, cv2.COLOR_RGB2BGR)

    # Feathered mask blend
    mask_float = skin_mask.astype(np.float32) / 255.0
    k = max(31, (min(orig_h, orig_w) // 25) | 1)
    mask_float = cv2.GaussianBlur(mask_float, (k, k), 0)
    mask_3ch = np.stack([mask_float] * 3, axis=-1)

    result = bgr * (1 - mask_3ch) + bgr_new * mask_3ch
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-variant", default="512")
    args = parser.parse_args()

    print("=" * 50)
    print("DPR - NORMALIZE DIRECTIONAL LIGHT")
    print("=" * 50)

    device = torch.device("cuda")

    try:
        result = relight_normalized(args.input, args.checkpoint, args.model_variant, device)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), result)
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()