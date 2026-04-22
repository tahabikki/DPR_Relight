"""
PyTorch Dataset for DPR Passport Photo Fine-tuning

This module provides PassportRelightDataset, a PyTorch Dataset that:
1. Loads paired (input, target) images from split dataset folders
2. Estimates SH (Spherical Harmonics) lighting coefficients from target images
3. Applies augmentations suitable for passport photos (horizontal flip, slight rotation)
4. Returns (input_tensor, target_tensor, sh_target) tuples for training

The key adaptation from original DPR:
- Original DPR: trained with (input_image, target_SH) → relit_output
- This variant: trains with (input_image, estimated_SH_from_target) → output,
  where loss compares output pixel-by-pixel with target_image

Modified from: Original DPR codebase
Changes: Added passport-specific augmentation, SH extraction from target images,
         PyTorch Dataset wrapper for ease of use with DataLoader
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional
import torchvision.transforms.functional as TF


class PassportRelightDataset(Dataset):
    """
    PyTorch Dataset for paired passport photo relighting.
    
    Loads images from split/{train,eval,test}/{input,target}/ directories
    and returns (input_image, target_image, sh_coefficients) tuples.
    """
    
    def __init__(
        self,
        split_dir: str,
        split_name: str = "train",
        image_size: int = 512,
        enable_augmentation: bool = True,
        horizontal_flip_prob: float = 0.3,
        max_rotation_deg: float = 5.0,
        sh_extraction_method: str = "diffuse",
        sh_clip_min: float = -1.0,
        sh_clip_max: float = 1.0,
    ):
        """
        Initialize the dataset.
        
        Args:
            split_dir: Path to dataset_split directory
            split_name: One of 'train', 'eval', 'test'
            image_size: Target size for resizing (height and width)
            enable_augmentation: Whether to apply augmentations
            horizontal_flip_prob: Probability of horizontal flip (0-1)
            max_rotation_deg: Maximum rotation angle in degrees
            sh_extraction_method: Method for SH estimation ('diffuse' or 'intrinsic')
            sh_clip_min, sh_clip_max: Clipping range for SH coefficients
        """
        self.split_dir = Path(split_dir)
        self.split_name = split_name
        self.image_size = image_size
        self.enable_augmentation = enable_augmentation
        self.horizontal_flip_prob = horizontal_flip_prob
        self.max_rotation_deg = max_rotation_deg
        self.sh_extraction_method = sh_extraction_method
        self.sh_clip_min = sh_clip_min
        self.sh_clip_max = sh_clip_max
        
        # Paths
        self.input_dir = self.split_dir / split_name / "input"
        self.target_dir = self.split_dir / split_name / "target"
        
        # Validate directories exist
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        if not self.target_dir.exists():
            raise FileNotFoundError(f"Target directory not found: {self.target_dir}")
        
        # Load file list (sort to ensure deterministic order)
        self.input_files = sorted(list(self.input_dir.glob("*.[jJ][pP][gG]")))
        self.target_files = sorted(list(self.target_dir.glob("*.[jJ][pP][gG]")))
        
        if len(self.input_files) == 0:
            raise FileNotFoundError(f"No input images found in {self.input_dir}")
        if len(self.target_files) == 0:
            raise FileNotFoundError(f"No target images found in {self.target_dir}")
        
        # Verify pairing
        input_stems = {f.stem for f in self.input_files}
        target_stems = {f.stem for f in self.target_files}
        if input_stems != target_stems:
            raise ValueError("Input and target images don't match! Check for orphan files.")
        
        # Create mapping: input_file -> target_file
        self.paired_files = []
        for input_file in self.input_files:
            # Find matching target (same stem, possibly different extension)
            target_matches = list(self.target_dir.glob(f"{input_file.stem}.*"))
            if target_matches:
                self.paired_files.append((input_file, target_matches[0]))
        
        if len(self.paired_files) == 0:
            raise ValueError("No valid pairs found after matching stems")
    
    def __len__(self) -> int:
        return len(self.paired_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            (input_tensor, target_tensor, sh_coefficients)
            - input_tensor: [1, H, W] (grayscale L channel from LAB)
            - target_tensor: [3, H, W] (RGB color image)
            - sh_coefficients: [9, 1, 1] (SH lighting vector)
        """
        input_path, target_path = self.paired_files[idx]
        
        # Load images
        input_img = cv2.imread(str(input_path))  # BGR
        target_img = cv2.imread(str(target_path))  # BGR
        
        if input_img is None:
            raise RuntimeError(f"Failed to load input image: {input_path}")
        if target_img is None:
            raise RuntimeError(f"Failed to load target image: {target_path}")
        
        # Convert BGR to RGB for consistency
        input_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        target_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        input_rgb = cv2.resize(input_rgb, (self.image_size, self.image_size))
        target_rgb = cv2.resize(target_rgb, (self.image_size, self.image_size))
        
        # Apply augmentations
        if self.enable_augmentation:
            input_rgb, target_rgb = self._apply_augmentation(input_rgb, target_rgb)
        
        # Extract L channel (luminance) from input image for the network input
        input_lab = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2LAB)
        input_l = input_lab[:, :, 0].astype(np.float32) / 255.0
        
        # Extract SH coefficients from target image
        sh_target = self._extract_sh_coefficients(target_rgb)
        
        # Convert to tensors
        input_tensor = torch.from_numpy(input_l[np.newaxis, ...]).float()  # [1, H, W]
        
        target_rgb_normalized = target_rgb.astype(np.float32) / 255.0
        target_tensor = torch.from_numpy(
            target_rgb_normalized.transpose(2, 0, 1)  # [H, W, 3] -> [3, H, W]
        ).float()
        
        sh_tensor = torch.from_numpy(sh_target).float()  # [9]
        sh_tensor = sh_tensor.reshape(9, 1, 1)  # [9, 1, 1] for broadcasting
        
        return input_tensor, target_tensor, sh_tensor
    
    def _apply_augmentation(
        self,
        input_rgb: np.ndarray,
        target_rgb: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentations safe for passport photos:
        - Horizontal flip (preserves frontal face orientation)
        - Small rotation (±5 degrees to preserve passport-style pose)
        - No color jitter (lighting is the feature being learned)
        """
        # Horizontal flip
        if np.random.rand() < self.horizontal_flip_prob:
            input_rgb = cv2.flip(input_rgb, 1)
            target_rgb = cv2.flip(target_rgb, 1)
        
        # Small rotation
        if self.max_rotation_deg > 0:
            angle = np.random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
            h, w = input_rgb.shape[:2]
            center = (w // 2, h // 2)
            
            # Get rotation matrix
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Apply rotation with border fill (replicate edges)
            input_rgb = cv2.warpAffine(
                input_rgb, M, (w, h),
                borderMode=cv2.BORDER_REPLICATE
            )
            target_rgb = cv2.warpAffine(
                target_rgb, M, (w, h),
                borderMode=cv2.BORDER_REPLICATE
            )
        
        return input_rgb, target_rgb
    
    def _extract_sh_coefficients(self, target_rgb: np.ndarray) -> np.ndarray:
        """
        Extract SH lighting coefficients from target image.
        
        Method 1 (used here - 'diffuse'):
            Simple approach: estimate SH from overall image brightness distribution.
            Assumes neutral face with diffuse reflectance.
            Robust to variations in albedo and geometry.
        
        Method 2 ('intrinsic' - not implemented):
            More complex: decompose image into shading and reflectance layers,
            then solve for SH from shading layer.
            Requires accurate normal map (typically needs face alignment/3D fitting).
        
        Args:
            target_rgb: [H, W, 3] RGB image (0-255)
        
        Returns:
            sh: [9] SH coefficient vector
        """
        if self.sh_extraction_method == "diffuse":
            return self._extract_sh_diffuse(target_rgb)
        elif self.sh_extraction_method == "intrinsic":
            # Placeholder for intrinsic decomposition (more complex)
            # For now, fall back to diffuse
            return self._extract_sh_diffuse(target_rgb)
        else:
            raise ValueError(f"Unknown SH extraction method: {self.sh_extraction_method}")
    
    def _extract_sh_diffuse(self, target_rgb: np.ndarray) -> np.ndarray:
        """
        Improved SH extraction for directional lighting.
        
        Approach:
        1. Convert to grayscale (luminance)
        2. Analyze shading gradients to estimate light direction
        3. Detect shadow regions
        4. Fit SH coefficients for directional light
        """
        # Convert to luminance
        gray = cv2.cvtColor(target_rgb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        # Compute gradients in different regions
        h, w = gray.shape
        grad_x = np.gradient(gray, axis=1)  # horizontal gradient
        grad_y = np.gradient(gray, axis=0)  # vertical gradient
        
        # Estimate light direction from gradient field
        # Positive grad_x = light from right
        # Negative grad_x = light from left
        # Positive grad_y = light from below
        # Negative grad_y = light from above
        mean_grad_x = np.mean(grad_x)
        mean_grad_y = np.mean(grad_y)
        
        # Detect shadow: dark regions with smooth gradients
        shadow_mask = (gray < 0.3).astype(np.float32)
        shadow_intensity = np.mean(shadow_mask)
        
        # Compute statistics
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Initialize SH coefficients
        sh = np.zeros(9, dtype=np.float32)
        
        # DC term: overall brightness
        sh[0] = (mean_brightness - 0.5) * 2.0
        
        # First-order terms (directional light)
        # Map gradients to SH first-order coefficients
        # SH[1] = Y (vertical, positive = light from above)
        # SH[2] = Z (depth, positive = toward camera)
        # SH[3] = X (horizontal, positive = light from right)
        sh[1] = np.clip(-mean_grad_y * 5, -1, 1)  # Y: vertical component
        sh[2] = 0.0  # Z: default
        sh[3] = np.clip(mean_grad_x * 5, -1, 1)  # X: horizontal component
        
        # Second-order terms for ambient
        sh[4] = mean_grad_x * mean_grad_y * 0.5  # YX
        sh[5] = -mean_grad_y * 0.2  # YZ
        sh[6] = std_brightness * 0.3  # 3Z^2 - 1
        sh[7] = mean_grad_x * 0.2  # XZ
        sh[8] = (mean_grad_x**2 - mean_grad_y**2) * 0.2  # X^2 - Y^2
        
        # Adjust for shadows: reduce first-order terms where shadows detected
        if shadow_intensity > 0.1:
            shadow_factor = 1.0 - shadow_intensity * 0.3
            sh[1] *= shadow_factor
            sh[3] *= shadow_factor
        
        # Clip to valid range
        sh = np.clip(sh, self.sh_clip_min, self.sh_clip_max)
        
        return sh


def create_dataloaders(
    split_dir: str,
    batch_size: int = 8,
    eval_batch_size: int = 16,
    num_workers: int = 0,
    image_size: int = 512,
    enable_augmentation: bool = True,
    seed: int = 42,
    device: str = "cpu",
    **kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and eval dataloaders (test is handled by infer.py).
    
    Args:
        split_dir: Path to dataset_split directory
        batch_size: Batch size for training
        eval_batch_size: Batch size for evaluation
        num_workers: Number of worker threads for DataLoader
        image_size: Target image size
        enable_augmentation: Whether to augment training data
        seed: Random seed for reproducibility
        device: Device type ('cpu' or 'cuda') - pin_memory only used for CUDA
        **kwargs: Additional arguments for PassportRelightDataset
    
    Returns:
        (train_loader, eval_loader)
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Only pin memory if using CUDA (CPU doesn't need it and causes warnings)
    use_pin_memory = (device.lower() == "cuda" and num_workers > 0)
    
    # Create datasets
    train_dataset = PassportRelightDataset(
        split_dir=split_dir,
        split_name="train",
        image_size=image_size,
        enable_augmentation=enable_augmentation,
        **kwargs
    )
    
    eval_dataset = PassportRelightDataset(
        split_dir=split_dir,
        split_name="eval",
        image_size=image_size,
        enable_augmentation=False,
        **kwargs
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=use_pin_memory
)
    
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    return train_loader, eval_loader


if __name__ == "__main__":
    # Quick test
    import argparse
    
    parser = argparse.ArgumentParser(description="Test PassportRelightDataset")
    parser.add_argument("--split-dir", type=str, default="dataset_split")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    
    print("Testing PassportRelightDataset...")
    
    try:
        train_loader, eval_loader, test_loader = create_dataloaders(
            split_dir=args.split_dir,
            batch_size=args.batch_size,
            num_workers=0
        )
        
        print(f"✓ Train loader: {len(train_loader)} batches")
        print(f"✓ Eval loader: {len(eval_loader)} batches")
        print(f"✓ Test loader: {len(test_loader)} batches")
        
        # Get first batch
        input_img, target_img, sh = next(iter(train_loader))
        print(f"\n✓ Sample batch:")
        print(f"  Input shape: {input_img.shape}")
        print(f"  Target shape: {target_img.shape}")
        print(f"  SH shape: {sh.shape}")
        print(f"  SH values: {sh[0].flatten()}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
