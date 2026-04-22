#!/usr/bin/env python3
"""
Inference Script for Fine-tuned DPR Model

Takes a single input passport photo and produces a relit/shadow-removed version
using the fine-tuned DPR model.

Usage:
    python scripts/infer.py --checkpoint checkpoints/best_model.pth --input input.jpg --output output.jpg

Modified from: Original DPR codebase
Changes: Single-image inference wrapper, checkpoint loading, path handling
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple
import torch
import cv2
import numpy as np

# Add parent directories to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "model"))
sys.path.insert(0, str(project_root / "utils"))


def detect_device() -> Tuple[torch.device, str]:
    """
    Automatically detect the best available device across all platforms.
    
    Checks for (in order):
    - Metal Performance Shaders (Apple Silicon/Mac)
    - CUDA (NVIDIA GPUs)
    - ROCm (AMD GPUs)
    - CPU (fallback)
    
    Returns:
        (torch.device, device_info_string)
    """
    # Try Metal (Apple Silicon & Mac)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            test_tensor = torch.randn(1, device='mps')
            device = torch.device('mps')
            return device, "🍎 Metal (Apple GPU)"
        except Exception:
            pass
    
    # Try CUDA (NVIDIA)
    if torch.cuda.is_available():
        try:
            cuda_device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            return cuda_device, f"🎮 CUDA (NVIDIA GPU: {gpu_name}, {gpu_mem:.2f} GB)"
        except Exception:
            pass
    
    # Try ROCm (AMD GPUs)
    if torch.cuda.is_available() and 'rocm' in torch.version.cuda.lower():
        try:
            device = torch.device('cuda')
            return device, "🔴 ROCm (AMD GPU)"
        except Exception:
            pass
    
    # Fallback to CPU
    return torch.device('cpu'), "💻 CPU (No GPU detected)"


def load_model(checkpoint_path: str, model_variant: str, device: torch.device):
    """Load fine-tuned model from checkpoint."""
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"📦 Loading architecture for {model_variant}x{model_variant}...")
    
    # Import and instantiate model
    if model_variant == "512":
        from defineHourglass_512_gray_skip import HourglassNet
        model = HourglassNet()
    elif model_variant == "1024":
        from defineHourglass_512_gray_skip import HourglassNet
        from defineHourglass_1024_gray_skip_matchFeature import HourglassNet_1024
        model_512 = HourglassNet()
        model = HourglassNet_1024(model_512, 16)
    else:
        raise ValueError(f"Unknown model variant: {model_variant}")
    
    # Load checkpoint
    print(f"📥 Loading checkpoint: {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle both fine-tuning checkpoint and raw state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def get_fixed_sh() -> np.ndarray:
    """
    Return fixed flat passport lighting SH coefficients.
    
    This creates even, flattering lighting without directional shadows.
    """
    sh = np.zeros(9, dtype=np.float32)
    sh[0] = 0.7   # Bright but not blown out
    sh[1] = 0.0   # No vertical gradient
    sh[2] = 0.0   # No depth gradient
    sh[3] = 0.0   # No horizontal gradient
    sh[4] = 0.0
    sh[5] = 0.0
    sh[6] = 0.1
    sh[7] = 0.0
    sh[8] = 0.0
    return sh


def estimate_sh_from_image(image_rgb: np.ndarray) -> np.ndarray:
    """
    Legacy function - now uses fixed flat SH instead of estimation.
    Kept for backward compatibility.
    """
    return get_fixed_sh()


def infer_single_image(
    image_path: str,
    checkpoint_path: str,
    model_variant: str = "512",
    device_str: str = "auto",
    output_size: int = 512
) -> np.ndarray:
    """
    Run inference on a single image.
    
    Args:
        image_path: Path to input passport photo
        checkpoint_path: Path to fine-tuned checkpoint
        model_variant: "512" or "1024"
        device_str: "auto", "cuda", or "cpu"
        output_size: Output resolution (512 or 1024)
    
    Returns:
        relit_image: [H, W, 3] BGR output image (0-255)
    """
    # Determine device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    # Load image
    print(f"\n📷 Loading input image: {image_path}")
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found or failed to load: {image_path}")
    
    orig_h, orig_w = image_bgr.shape[:2]
    print(f"   Original size: {orig_w}x{orig_h}")
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    image_rgb_resized = cv2.resize(image_rgb, (output_size, output_size))
    print(f"   Resized to: {output_size}x{output_size}")
    
    # Extract L, a, b channels from input (keep a, b for later recombination)
    image_lab = cv2.cvtColor(image_rgb_resized, cv2.COLOR_RGB2LAB)
    input_l = image_lab[:, :, 0].astype(np.float32) / 255.0
    input_a = image_lab[:, :, 1].astype(np.float32)  # Keep a channel
    input_b = image_lab[:, :, 2].astype(np.float32)  # Keep b channel
    
    # Use fixed flat SH (passport lighting)
    print(f"\n💡 Using fixed flat passport lighting")
    sh = get_fixed_sh()
    print(f"   SH coefficients: {sh}")
    
    # Convert to tensors (L channel only)
    input_tensor = torch.from_numpy(input_l[np.newaxis, ...]).float().to(device)  # [1, 1, H, W]
    sh_tensor = torch.from_numpy(sh).float().reshape(1, 9, 1, 1).to(device)  # [1, 9, 1, 1]
    
    # Load model
    print(f"\n🧠 Loading model...")
    model = load_model(checkpoint_path, model_variant, device)
    
    # Run inference
    print(f"🚀 Running inference...")
    with torch.no_grad():
        output_img, output_sh = model(input_tensor, sh_tensor, 0)
    
    # Extract output L channel
    output_l = output_img[0, 0].cpu().numpy()  # [H, W]
    output_l = np.clip(output_l * 255, 0, 255).astype(np.uint8)
    
    # Recombine with original a, b channels
    output_lab = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    output_lab[:, :, 0] = output_l
    output_lab[:, :, 1] = input_a.astype(np.uint8)
    output_lab[:, :, 2] = input_b.astype(np.uint8)
    
    # Convert LAB back to RGB
    output_rgb = cv2.cvtColor(output_lab, cv2.COLOR_LAB2RGB)
    
    # Convert back to original size
    if (orig_h, orig_w) != (output_size, output_size):
        print(f"   Resizing output back to: {orig_w}x{orig_h}")
        output_rgb = cv2.resize(output_rgb, (orig_w, orig_h))
    
    # Convert to BGR for OpenCV
    output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
    
    return output_bgr


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a single passport photo using fine-tuned DPR"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to fine-tuned checkpoint (e.g., checkpoints/best_model.pth)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input passport photo"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save output relit image"
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        default="512",
        choices=["512", "1024"],
        help="Model variant (512 or 1024)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps", "rocm"],
        help="Device to use: auto (auto-detect), cuda (NVIDIA), mps (Apple), rocm (AMD), cpu"
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=512,
        help="Output resolution (512 or 1024)"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("DPR INFERENCE - PASSPORT PHOTO RELIGHTING")
    print(f"{'='*70}")
    
    # Determine device with automatic detection
    if args.device == "auto":
        device, device_info = detect_device()
        print(f"🔍 Auto-detecting device...")
        print(f"   {device_info}")
    else:
        if args.device == "mps":
            device = torch.device("mps")
            device_info = "🍎 Metal (Apple GPU) - manual"
        elif args.device == "cuda":
            device = torch.device("cuda")
            device_info = "🎮 CUDA (NVIDIA GPU) - manual"
        elif args.device == "rocm":
            device = torch.device("cuda")
            device_info = "🔴 ROCm (AMD GPU) - manual"
        else:
            device = torch.device("cpu")
            device_info = "💻 CPU - manual"
        
        print(f"📌 Device (manual): {device_info}")
    
    # Run inference
    try:
        output_image = infer_single_image(
            image_path=args.input,
            checkpoint_path=args.checkpoint,
            model_variant=args.model_variant,
            device_str=str(device),
            output_size=args.output_size
        )
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), output_image)
    print(f"\n✅ Output saved to: {output_path}")
    
    print(f"\n💡 Tips for best results:")
    print(f"   - Input should be a close-up frontal face (like a passport photo)")
    print(f"   - Ensure adequate lighting in the input for best relighting")
    print(f"   - Output will have more even illumination and reduced shadows")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
