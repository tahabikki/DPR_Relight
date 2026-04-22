#!/usr/bin/env python3
"""
Fine-tuning Training Script for DPR Passport Photo Relighting

This script trains (fine-tunes) a pretrained DPR model on paired passport photos.

Usage:
    python scripts/train.py --config configs/finetune_passport.yaml
    python scripts/train.py --config configs/finetune_passport.yaml --device cuda

Modified from: Original DPR codebase
Changes: Added PyTorch training loop, checkpoint management, metrics logging,
         YAML config support, and Windows path compatibility
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directories to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "model"))
sys.path.insert(0, str(project_root / "utils"))

from data.dataset import create_dataloaders


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model(config: Dict, device: torch.device):
    """
    Load pretrained model and configure for fine-tuning.
    
    Args:
        config: Configuration dictionary
        device: torch.device (cuda or cpu)
    
    Returns:
        (model, device)
    """
    model_variant = config['model']['variant']
    pretrained_path = Path(config['checkpoint']['pretrained_checkpoint'])
    
    if not pretrained_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")
    
    print(f"\n📦 Loading architecture for {model_variant}x{model_variant}...")
    
    # Import and instantiate appropriate model
    if model_variant == "512":
        from defineHourglass_512_gray_skip import HourglassNet
        model = HourglassNet()
        checkpoint_path = pretrained_path
    elif model_variant == "1024":
        from defineHourglass_512_gray_skip import HourglassNet
        from defineHourglass_1024_gray_skip_matchFeature import HourglassNet_1024
        
        model_512 = HourglassNet()
        model = HourglassNet_1024(model_512, config['model']['base_channels'])
        checkpoint_path = pretrained_path
    else:
        raise ValueError(f"Unknown model variant: {model_variant}")
    
    # Load pretrained weights
    print(f"📥 Loading pretrained weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to load state dict, handling mismatched layers gracefully
    try:
        incompatible_keys = model.load_state_dict(checkpoint, strict=False)
        if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
            print(f"   ⚠️  Skipped mismatched layers (expected for output layer change):")
            if incompatible_keys.missing_keys:
                for key in incompatible_keys.missing_keys[:3]:
                    print(f"      Missing: {key}")
            if incompatible_keys.unexpected_keys:
                for key in incompatible_keys.unexpected_keys[:3]:
                    print(f"      Unexpected: {key}")
        else:
            print(f"   ✅ All layers loaded successfully")
    except RuntimeError as e:
        print(f"   ⚠️  Could not load with strict=False: {str(e)[:100]}...")
        print(f"   💡 Loading compatible layers manually...")
        
        # Load layers manually, skipping ones that don't match
        model_state = model.state_dict()
        incompatible_count = 0
        loaded_count = 0
        
        for key, value in checkpoint.items():
            if key in model_state:
                if model_state[key].shape == value.shape:
                    model_state[key] = value
                    loaded_count += 1
                else:
                    incompatible_count += 1
                    if incompatible_count <= 3:
                        print(f"      Skipped {key}: shape {value.shape} vs {model_state[key].shape}")
        
        model.load_state_dict(model_state)
        print(f"   ✅ Loaded {loaded_count} layers, skipped {incompatible_count} mismatched")
    
    # Move to device
    model = model.to(device)
    
    # Optionally freeze encoder
    if config['model']['freeze_encoder']:
        print("🔒 Freezing encoder layers...")
        freeze_encoder(model)
    
    return model


def freeze_encoder(model):
    """
    Freeze encoder weights (Hourglass downsampling path).
    
    This is a heuristic: we freeze layers up to the middle of the network.
    For fine-tuning on limited data, this reduces overfitting risk.
    """
    # Get all named parameters
    encoder_patterns = ['down', 'maxpool']  # Common patterns in encoder
    
    trainable_count = 0
    frozen_count = 0
    
    for name, param in model.named_parameters():
        # Check if this is likely part of the encoder
        is_encoder = any(pattern.lower() in name.lower() for pattern in encoder_patterns)
        
        if is_encoder:
            param.requires_grad = False
            frozen_count += 1
        else:
            trainable_count += 1
    
    print(f"   Frozen layers: {frozen_count}, Trainable layers: {trainable_count}")


def setup_optimizer(model, config: Dict):
    """
    Set up optimizer and learning rate schedule.
    
    Args:
        model: PyTorch model
        config: Configuration dictionary
    
    Returns:
        optimizer
    """
    # Get only trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    
    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    
    print(f"📊 Optimizer: Adam")
    print(f"   Learning rate: {lr}")
    print(f"   Weight decay: {weight_decay}")
    print(f"   Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    return optimizer


def get_loss_fn(config: Dict):
    """Get loss function based on config."""
    loss_type = config['loss']['reconstruction_loss']
    
    if loss_type == 'l1':
        return nn.L1Loss()
    elif loss_type == 'l2':
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def train_one_epoch(
    model,
    train_loader: DataLoader,
    optimizer,
    loss_fn,
    device: torch.device,
    epoch: int,
    config: Dict
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        optimizer: Optimizer
        loss_fn: Loss function
        device: torch.device
        epoch: Current epoch number
        config: Configuration dictionary
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", ncols=80)
    
    for batch_idx, (input_img, target_img, sh_target) in enumerate(pbar):
        # Move to device
        input_img = input_img.to(device)
        target_img = target_img.to(device)
        sh_target = sh_target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output_img, output_sh = model(input_img, sh_target, 0)  # 0 = training mode
        
        # Compute loss
        loss = loss_fn(output_img, target_img)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        
        # Update progress bar
        avg_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
    
    avg_epoch_loss = total_loss / len(train_loader)
    return avg_epoch_loss


def eval_one_epoch(
    model,
    eval_loader: DataLoader,
    loss_fn,
    device: torch.device,
    epoch: int,
    config: Dict
) -> float:
    """
    Evaluate for one epoch (no gradients).
    
    Args:
        model: PyTorch model
        eval_loader: DataLoader for evaluation data
        loss_fn: Loss function
        device: torch.device
        epoch: Current epoch number
        config: Configuration dictionary
    
    Returns:
        Average loss for the epoch
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        pbar = tqdm(eval_loader, desc=f"Epoch {epoch+1} [Eval]", ncols=80)
        
        for batch_idx, (input_img, target_img, sh_target) in enumerate(pbar):
            # Move to device
            input_img = input_img.to(device)
            target_img = target_img.to(device)
            sh_target = sh_target.to(device)
            
            # Forward pass
            output_img, output_sh = model(input_img, sh_target, 0)
            
            # Compute loss
            loss = loss_fn(output_img, target_img)
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
    
    avg_epoch_loss = total_loss / len(eval_loader)
    return avg_epoch_loss


def save_checkpoint(model, optimizer, epoch: int, loss: float, save_path: Path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, save_path)


def plot_training_curve(train_losses, eval_losses, save_path: Path):
    """Plot and save training curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o', markersize=4)
    plt.plot(eval_losses, label='Eval Loss', marker='s', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves: DPR Fine-tuning for Passport Photos')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 Training curve saved to {save_path}")


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
            # Test if MPS actually works
            test_tensor = torch.randn(1, device='mps')
            device = torch.device('mps')
            return device, "🍎 Metal (Apple GPU)"
        except Exception:
            pass  # Metal available but not working, try next
    
    # Try CUDA (NVIDIA)
    if torch.cuda.is_available():
        try:
            cuda_device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            return cuda_device, f"🎮 CUDA (NVIDIA GPU: {gpu_name}, {gpu_mem:.2f} GB)"
        except Exception:
            pass  # CUDA available but not working, try next
    
    # Try ROCm (AMD GPUs)
    if torch.cuda.is_available() and 'rocm' in torch.version.cuda.lower():
        try:
            device = torch.device('cuda')  # ROCm also uses 'cuda' device string
            return device, "🔴 ROCm (AMD GPU)"
        except Exception:
            pass  # ROCm available but not working, try next
    
    # Fallback to CPU
    return torch.device('cpu'), "💻 CPU (No GPU detected)"


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune DPR for passport photo relighting"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/finetune_passport.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps", "rocm"],
        help="Device to use: auto (auto-detect), cuda (NVIDIA), mps (Apple), rocm (AMD), cpu"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"\n{'='*70}")
    print("DPR FINE-TUNING FOR PASSPORT PHOTO RELIGHTING")
    print(f"{'='*70}\n")
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    print(f"📄 Loading config from {config_path}...")
    config = load_config(config_path)
    
    # Determine device with automatic detection
    if args.device == "auto":
        device, device_info = detect_device()
        print(f"🔍 Auto-detecting device...")
        print(f"   {device_info}")
    else:
        # Manual device selection
        if args.device == "mps":
            device = torch.device("mps")
            device_info = "🍎 Metal (Apple GPU) - manual"
        elif args.device == "cuda":
            device = torch.device("cuda")
            device_info = "🎮 CUDA (NVIDIA GPU) - manual"
        elif args.device == "rocm":
            device = torch.device("cuda")  # ROCm uses cuda device string
            device_info = "🔴 ROCm (AMD GPU) - manual"
        else:  # cpu
            device = torch.device("cpu")
            device_info = "💻 CPU - manual"
        
        print(f"📌 Device (manual): {device_info}")
    
    # Set random seed
    seed = config['training']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"🌱 Random seed: {seed}")
    
    # Create output directories
    save_dir = Path(config['checkpoint']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Check dataset exists
    split_dataset_path = Path(config['data']['split_dataset_path'])
    if not split_dataset_path.exists():
        print(f"\n❌ Split dataset not found: {split_dataset_path}")
        print(f"   Please run: python data/prepare_splits.py")
        return False
    
    print(f"\n📂 Dataset: {split_dataset_path}")
    
    # Load model
    try:
        model = setup_model(config, device)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        return False
    
    # Create dataloaders
    print(f"\n📊 Creating dataloaders...")
    try:
        train_loader, eval_loader, test_loader = create_dataloaders(
            split_dir=str(split_dataset_path),
            batch_size=config['training']['batch_size'],
            eval_batch_size=config['training']['eval_batch_size'],
            num_workers=config['training']['num_workers'],
            image_size=config['data']['image_size'],
            enable_augmentation=True,
            seed=seed,
            device=device.type,  # Pass device type to disable pin_memory on CPU
            horizontal_flip_prob=config['data']['augmentation']['horizontal_flip_prob'],
            max_rotation_deg=config['data']['augmentation']['max_rotation_deg'],
            sh_extraction_method=config['lighting']['extraction_method'],
            sh_clip_min=config['lighting']['sh_clip_min'],
            sh_clip_max=config['lighting']['sh_clip_max']
        )
    except Exception as e:
        print(f"\n❌ Error creating dataloaders: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"   Train: {len(train_loader)} batches")
    print(f"   Eval: {len(eval_loader)} batches")
    print(f"   Test: {len(test_loader)} batches")
    
    # Setup optimizer and loss
    optimizer = setup_optimizer(model, config)
    loss_fn = get_loss_fn(config)
    print(f"   Loss: {config['loss']['reconstruction_loss'].upper()}")
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    train_losses = []
    eval_losses = []
    best_eval_loss = float('inf')
    best_epoch = -1
    
    print(f"\n🚀 Starting fine-tuning for {num_epochs} epochs...")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch, config
        )
        train_losses.append(train_loss)
        
        # Eval
        eval_loss = eval_one_epoch(
            model, eval_loader, loss_fn, device, epoch, config
        )
        eval_losses.append(eval_loss)
        
        # Print summary
        print(f"\n[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.6f} | Eval Loss: {eval_loss:.6f}")
        
        # Save checkpoint
        if config['checkpoint']['save_best'] and eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_epoch = epoch
            best_checkpoint_path = save_dir / "best_model.pth"
            save_checkpoint(model, optimizer, epoch, eval_loss, best_checkpoint_path)
            print(f"✅ Best checkpoint saved: {best_checkpoint_path}")
        
        if config['checkpoint']['save_interval'] > 0 and (epoch + 1) % config['checkpoint']['save_interval'] == 0:
            checkpoint_path = save_dir / f"checkpoint_epoch_{epoch+1:03d}.pth"
            save_checkpoint(model, optimizer, epoch, eval_loss, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    elapsed_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✅ Training complete!")
    print(f"   Time elapsed: {elapsed_time/60:.1f} minutes")
    print(f"   Best epoch: {best_epoch + 1} with eval loss: {best_eval_loss:.6f}")
    
    # Plot and save training curves
    curve_path = save_dir / "training_curve.png"
    plot_training_curve(train_losses, eval_losses, curve_path)
    
    # Save final checkpoint
    final_checkpoint_path = save_dir / "final_model.pth"
    save_checkpoint(model, optimizer, num_epochs - 1, eval_losses[-1], final_checkpoint_path)
    print(f"💾 Final model saved: {final_checkpoint_path}")
    
    print(f"\n📂 All outputs saved to: {save_dir}")
    print(f"\n💡 Next steps:")
    print(f"   1. Evaluate on test set: python scripts/eval.py --checkpoint {best_checkpoint_path}")
    print(f"   2. Run inference: python scripts/infer.py --checkpoint {best_checkpoint_path} --input <image_path>")
    
    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
