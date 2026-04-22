"""
PyTorch Dataset for DPR Passport Photo Fine-tuning (FIXED VERSION)

KEY FIX (vs previous broken version):
====================================
The previous version tried to make DPR predict RGB color from L channel input
using fake SH coefficients. This was architecturally wrong and produced artifacts.

This version stays faithful to DPR's original design:
  - Input : L channel of the poorly-lit photo
  - Target: L channel of the well-lit photo (same person, paired)
  - SH    : FIXED flat passport lighting (not estimated from image)

NEW: Skin mask support for focused training:
  - Uses color-based skin detection (YCrCb + HSV + RGB)
  - Model trains ONLY on skin pixels (not background)
  - Better results because model focuses on skin lighting only
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple


# ---------------------------------------------------------------------------
# Fixed flat passport-style target lighting (Spherical Harmonics, 9 coefficients)
#
# SH[0]      : DC / ambient          -> large positive = bright overall
# SH[1..3]   : first-order direction -> zero = no directional light
# SH[4..8]   : second-order          -> zero = no complex shading
#
# This vector says: "uniform frontal ambient light, no shadows, no direction".
# Using the SAME vector for every sample teaches the model ONE target style,
# which is exactly what we want for consistent passport output.
# ---------------------------------------------------------------------------
FLAT_PASSPORT_SH = np.array(
    [0.70, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Skin detection for training mask (same as utils/skin_mask.py)
# ---------------------------------------------------------------------------
def create_skin_mask(image_bgr: np.ndarray, min_area: int = 3000) -> np.ndarray:
    """Create skin mask using color detection (YCrCb + HSV + RGB)."""
    h, w = image_bgr.shape[:2]

    # YCrCb skin detection
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    lower_ycrcb = np.array([0, 128, 65])
    upper_ycrcb = np.array([255, 180, 145])
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    # HSV skin detection
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([0, 15, 50])
    upper_hsv = np.array([20, 255, 255])
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # RGB skin rules
    b, g, r = cv2.split(image_bgr.astype(np.float32))
    mask_rgb = ((r > 95) & (g > 40) & (b > 20) &
              ((r - g) > 15) & ((r - b) > 15) &
              (r > g) & (r > b)).astype(np.uint8) * 255

    # Combine (union)
    mask = cv2.bitwise_or(mask_ycrcb, mask_hsv)
    mask = cv2.bitwise_or(mask, mask_rgb)

    # Cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Keep largest region only
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros((h, w), dtype=np.uint8)
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(clean_mask, [contour], -1, 255, -1)

    # Dilate and smooth
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.dilate(clean_mask, kernel, iterations=1)
    clean_mask = cv2.GaussianBlur(clean_mask, (9, 9), 0)

    return clean_mask


class PassportRelightDataset(Dataset):
    """
    Paired passport photo dataset for DPR fine-tuning.

    Each sample returns:
        input_L  : [1, H, W]  luminance of the bad-lighting photo (0..1)
        target_L : [1, H, W]  luminance of the good-lighting photo (0..1)
        sh       : [9, 1, 1]  fixed flat passport-lighting SH

    NEW: use_skin_mask=True trains ONLY on skin pixels!
    """

    def __init__(
        self,
        split_dir: str,
        split_name: str = "train",
        image_size: int = 512,
        enable_augmentation: bool = True,
        use_skin_mask: bool = True,  # NEW: only train on skin
        horizontal_flip_prob: float = 0.3,
        max_rotation_deg: float = 5.0,
        **_ignored,  # accept extra kwargs silently for backward compat
    ):
        self.split_dir = Path(split_dir)
        self.split_name = split_name
        self.image_size = image_size
        self.enable_augmentation = enable_augmentation
        self.use_skin_mask = use_skin_mask  # NEW
        self.horizontal_flip_prob = horizontal_flip_prob
        self.max_rotation_deg = max_rotation_deg

        self.input_dir = self.split_dir / split_name / "input"
        self.target_dir = self.split_dir / split_name / "target"

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        if not self.target_dir.exists():
            raise FileNotFoundError(f"Target directory not found: {self.target_dir}")

        # Gather paired files (match by stem)
        input_files = sorted(p for p in self.input_dir.iterdir()
                             if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        self.paired_files = []
        for f in input_files:
            matches = list(self.target_dir.glob(f"{f.stem}.*"))
            if matches:
                self.paired_files.append((f, matches[0]))

        if not self.paired_files:
            raise ValueError(
                f"No paired files found between {self.input_dir} and {self.target_dir}"
            )

    def __len__(self) -> int:
        return len(self.paired_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_path, target_path = self.paired_files[idx]

        # Load both images as RGB
        in_bgr = cv2.imread(str(input_path))
        tg_bgr = cv2.imread(str(target_path))
        if in_bgr is None or tg_bgr is None:
            raise RuntimeError(f"Failed to read pair: {input_path} / {target_path}")

        in_rgb = cv2.cvtColor(in_bgr, cv2.COLOR_BGR2RGB)
        tg_rgb = cv2.cvtColor(tg_bgr, cv2.COLOR_BGR2RGB)

        # Resize to model size
        in_rgb = cv2.resize(in_rgb, (self.image_size, self.image_size))
        tg_rgb = cv2.resize(tg_rgb, (self.image_size, self.image_size))

        # Augmentations (applied synchronously to both images)
        if self.enable_augmentation:
            in_rgb, tg_rgb = self._augment(in_rgb, tg_rgb)

        # Extract L channels
        in_L = cv2.cvtColor(in_rgb, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32) / 255.0
        tg_L = cv2.cvtColor(tg_rgb, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32) / 255.0

        # Apply skin mask (NEW: train ONLY on skin pixels)
        if self.use_skin_mask:
            # Create skin mask from input image
            skin_mask = create_skin_mask(in_rgb)
            skin_mask = skin_mask.astype(np.float32) / 255.0  # [0,1]

            # Apply mask to input and target
            in_L = in_L * skin_mask
            tg_L = tg_L * skin_mask

        # To tensors
        input_tensor = torch.from_numpy(in_L[None, ...]).float()     # [1, H, W]
        target_tensor = torch.from_numpy(tg_L[None, ...]).float()    # [1, H, W]
        sh_tensor = torch.from_numpy(FLAT_PASSPORT_SH).float().reshape(9, 1, 1)

        return input_tensor, target_tensor, sh_tensor

    # ------------------------------------------------------------------
    # Augmentation: horizontal flip + small rotation, applied identically
    # to input and target so the pair stays aligned.
    # ------------------------------------------------------------------
    def _augment(self, a: np.ndarray, b: np.ndarray):
        if np.random.rand() < self.horizontal_flip_prob:
            a = cv2.flip(a, 1)
            b = cv2.flip(b, 1)
        if self.max_rotation_deg > 0:
            angle = np.random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
            h, w = a.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            a = cv2.warpAffine(a, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            b = cv2.warpAffine(b, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return a, b


# ---------------------------------------------------------------------------
# DataLoader factory (same signature as before so train.py doesn't break)
# ---------------------------------------------------------------------------
def create_dataloaders(
    split_dir: str,
    batch_size: int = 8,
    eval_batch_size: int = 16,
    num_workers: int = 0,
    image_size: int = 512,
    enable_augmentation: bool = True,
    seed: int = 42,
    device: str = "cpu",
    **kwargs,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    use_pin_memory = (device.lower() == "cuda" and num_workers > 0)

    train_ds = PassportRelightDataset(
        split_dir=split_dir, split_name="train",
        image_size=image_size, enable_augmentation=enable_augmentation, **kwargs,
    )
    eval_ds = PassportRelightDataset(
        split_dir=split_dir, split_name="eval",
        image_size=image_size, enable_augmentation=False, **kwargs,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_pin_memory,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory,
    )
    return train_loader, eval_loader


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", default="dataset_split")
    args = parser.parse_args()

    train_loader, eval_loader = create_dataloaders(args.split_dir, batch_size=2)
    in_t, tg_t, sh_t = next(iter(train_loader))
    print(f"Input  : {in_t.shape}  range [{in_t.min():.3f}, {in_t.max():.3f}]")
    print(f"Target : {tg_t.shape}  range [{tg_t.min():.3f}, {tg_t.max():.3f}]")
    print(f"SH     : {sh_t.shape}  values {sh_t[0].flatten().tolist()}")