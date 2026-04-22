#!/usr/bin/env python3
"""
Evaluation Script for DPR Fine-tuned Model

Evaluates the fine-tuned model on the test/eval split and reports:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)

Also saves visual comparison grids (input | prediction | target).

Modified from: Original DPR codebase
Changes: Added PyTorch evaluation loop, metrics computation, visualization
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

# Add parent directories to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "model"))
sys.path.insert(0, str(project_root / "utils"))

from data.dataset import create_dataloaders


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
    
    # Handle both fine-tuning checkpoint (with 'model_state_dict') and raw state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute PSNR between prediction and target.
    
    Args:
        pred: [B, C, H, W] predicted images (0-1)
        target: [B, C, H, W] target images (0-1)
    
    Returns:
        PSNR value in dB
    """
    mse = F.mse_loss(pred, target, reduction='mean')
    if mse == 0:
        return 100.0
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute SSIM between prediction and target.
    
    Args:
        pred: [H, W, 3] predicted image (0-255)
        target: [H, W, 3] target image (0-255)
    
    Returns:
        SSIM value (0-1)
    """
    # Convert to uint8
    pred = np.clip(pred * 255, 0, 255).astype(np.uint8)
    target = np.clip(target * 255, 0, 255).astype(np.uint8)
    
    # Compute SSIM for each channel and average
    ssim_vals = []
    for c in range(3):
        # Use opencv for SSIM computation
        # Alternative: use skimage.metrics.structural_similarity
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        pred_c = pred[:, :, c].astype(np.float64)
        target_c = target[:, :, c].astype(np.float64)
        
        mean_pred = cv2.blur(pred_c, (11, 11))
        mean_target = cv2.blur(target_c, (11, 11))
        
        mean_pred_sq = cv2.blur(pred_c ** 2, (11, 11))
        mean_target_sq = cv2.blur(target_c ** 2, (11, 11))
        
        mean_pred_target = cv2.blur(pred_c * target_c, (11, 11))
        
        sigma_pred_sq = mean_pred_sq - mean_pred ** 2
        sigma_target_sq = mean_target_sq - mean_target ** 2
        sigma_pred_target = mean_pred_target - mean_pred * mean_target
        
        ssim = ((2 * mean_pred * mean_target + c1) * (2 * sigma_pred_target + c2)) / \
               ((mean_pred ** 2 + mean_target ** 2 + c1) * (sigma_pred_sq + sigma_target_sq + c2))
        
        ssim_vals.append(np.mean(ssim))
    
    return np.mean(ssim_vals)


def compute_lpips_simple(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Simple perceptual loss approximation using feature similarity.
    
    For real LPIPS, you'd need to use a pretrained network like VGG or AlexNet.
    This is a simplified approximation based on low-level features.
    
    Args:
        pred: [B, C, H, W] (0-1)
        target: [B, C, H, W] (0-1)
    
    Returns:
        LPIPS approximation (lower is better)
    """
    # Simple approximation: compute L2 distance at multiple scales
    total_dist = 0.0
    
    for scale in [1, 2, 4]:
        if scale > 1:
            pred_scaled = F.interpolate(pred, scale_factor=1/scale, mode='bilinear', align_corners=False)
            target_scaled = F.interpolate(target, scale_factor=1/scale, mode='bilinear', align_corners=False)
        else:
            pred_scaled = pred
            target_scaled = target
        
        # Compute L2 distance
        dist = torch.mean((pred_scaled - target_scaled) ** 2)
        total_dist += dist
    
    return (total_dist / 3).item()


def visualize_batch(input_imgs, pred_imgs, target_imgs, save_path: Path, num_samples: int = 16):
    """
    Create a grid visualization of input | prediction | target.
    
    Args:
        input_imgs: [B, 1, H, W] input images (0-1)
        pred_imgs: [B, 3, H, W] predicted images (0-1)
        target_imgs: [B, 3, H, W] target images (0-1)
        save_path: Path to save the grid
        num_samples: Number of samples to visualize
    """
    b = min(input_imgs.shape[0], num_samples)
    
    # Create figure with num_samples rows and 3 columns
    fig, axes = plt.subplots(b, 3, figsize=(12, 4 * b))
    if b == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(b):
        # Input (convert L channel to 3-channel for display)
        input_img = input_imgs[i, 0].cpu().numpy()
        input_img = np.stack([input_img] * 3, axis=-1)
        
        # Prediction
        pred_img = pred_imgs[i].cpu().numpy().transpose(1, 2, 0)
        
        # Target
        target_img = target_imgs[i].cpu().numpy().transpose(1, 2, 0)
        
        # Display
        axes[i, 0].imshow(input_img, cmap='gray')
        axes[i, 0].set_title(f"Input {i+1}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(np.clip(pred_img, 0, 1))
        axes[i, 1].set_title(f"Prediction {i+1}")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(np.clip(target_img, 0, 1))
        axes[i, 2].set_title(f"Target {i+1}")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"📊 Visualization saved to {save_path}")


def evaluate(
    model,
    dataloader,
    device: torch.device,
    split_name: str = "test",
    output_dir: Optional[Path] = None,
    num_visual_samples: int = 16
) -> Dict[str, float]:
    """
    Evaluate model on a dataloader.
    
    Args:
        model: Fine-tuned model
        dataloader: DataLoader for evaluation
        device: torch.device
        split_name: Name of split (for logging)
        output_dir: Directory to save visualizations
        num_visual_samples: Number of samples to visualize
    
    Returns:
        Dict with metrics
    """
    model.eval()
    
    psnr_vals = []
    ssim_vals = []
    lpips_vals = []
    
    all_pred = []
    all_target = []
    all_input = []
    
    print(f"\n📊 Evaluating on {split_name} split...")
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=split_name.upper(), ncols=80)
        
        for batch_idx, (input_img, target_img, sh_target) in enumerate(pbar):
            # Move to device
            input_img = input_img.to(device)
            target_img = target_img.to(device)
            sh_target = sh_target.to(device)
            
            # Forward pass
            pred_img, _ = model(input_img, sh_target, 0)
            
            # Clamp to valid range
            pred_img = torch.clamp(pred_img, 0, 1)
            
            # Compute metrics
            psnr = compute_psnr(pred_img, target_img)
            psnr_vals.append(psnr)
            
            # Convert to numpy for SSIM/LPIPS
            pred_np = pred_img.cpu().numpy().transpose(0, 2, 3, 1)  # [B, H, W, 3]
            target_np = target_img.cpu().numpy().transpose(0, 2, 3, 1)
            
            for b in range(pred_np.shape[0]):
                ssim = compute_ssim(pred_np[b], target_np[b])
                ssim_vals.append(ssim)
            
            lpips = compute_lpips_simple(pred_img, target_img)
            lpips_vals.append(lpips)
            
            # Store for visualization
            if batch_idx < 2:  # Keep first few batches for visualization
                all_input.append(input_img.cpu())
                all_pred.append(pred_img.cpu())
                all_target.append(target_img.cpu())
            
            pbar.set_postfix({
                'psnr': f"{np.mean(psnr_vals):.2f}",
                'ssim': f"{np.mean(ssim_vals):.4f}",
                'lpips': f"{np.mean(lpips_vals):.4f}"
            })
    
    # Compute averages
    avg_psnr = np.mean(psnr_vals)
    avg_ssim = np.mean(ssim_vals)
    avg_lpips = np.mean(lpips_vals)
    
    metrics = {
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'lpips': avg_lpips,
    }
    
    # Visualize if possible
    if output_dir and all_pred:
        all_input_cat = torch.cat(all_input, dim=0)
        all_pred_cat = torch.cat(all_pred, dim=0)
        all_target_cat = torch.cat(all_target, dim=0)
        
        viz_path = output_dir / f"eval_samples_{split_name}.png"
        visualize_batch(all_input_cat, all_pred_cat, all_target_cat, viz_path, num_visual_samples)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned DPR model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to fine-tuned checkpoint"
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        default="dataset_split",
        help="Path to split dataset"
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        default="512",
        choices=["512", "1024"],
        help="Model variant (512 or 1024)"
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default="test",
        choices=["train", "eval", "test"],
        help="Split to evaluate on"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps", "rocm"],
        help="Device to use: auto (auto-detect), cuda (NVIDIA), mps (Apple), rocm (AMD), cpu"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation"
    )
    
    args = parser.parse_args()
    
    # Determine device with automatic detection
    if args.device == "auto":
        device, device_info = detect_device()
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
    
    print(f"\n{'='*70}")
    print("DPR EVALUATION - PASSPORT PHOTO RELIGHTING")
    print(f"{'='*70}\n")
    print(f"📍 Device: {device_info}")
    print(f"📂 Dataset: {args.split_dir}")
    print(f"📊 Checkpoint: {args.checkpoint}")
    
    # Load model
    try:
        model = load_model(args.checkpoint, args.model_variant, device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Create dataloader
    try:
        _, eval_loader, test_loader = create_dataloaders(
            split_dir=args.split_dir,
            batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            num_workers=0,
            image_size=512 if args.model_variant == "512" else 1024,
            device=device.type  # Pass device type to disable pin_memory on CPU
        )
        
        loaders = {
            'eval': eval_loader,
            'test': test_loader
        }
        
        dataloader = loaders[args.eval_split]
        
    except Exception as e:
        print(f"❌ Error creating dataloader: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Evaluate
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = evaluate(
        model,
        dataloader,
        device,
        split_name=args.eval_split,
        output_dir=output_dir,
        num_visual_samples=16
    )
    
    # Print results
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS ({args.eval_split.upper()})")
    print(f"{'='*70}")
    print(f"PSNR:  {metrics['psnr']:.4f} dB")
    print(f"SSIM:  {metrics['ssim']:.4f}")
    print(f"LPIPS: {metrics['lpips']:.6f}")
    print(f"{'='*70}\n")
    
    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
