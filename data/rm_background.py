#!/usr/bin/env python3
"""
Background Removal Script using Rembg

Removes background from passport photos using AI-based background removal.
Creates a new dataset with transparent backgrounds (saved as PNG with alpha).

Usage:
    python data/rm_background.py --src dataset --dst dataset_no_bg

Or integrate with prepare_splits.py:
    python data/prepare_splits.py --src dataset --dst dataset_split --rm-bg
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import shutil

# Try to import rembg
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("[WARNING] rembg not installed. Install with: pip install rembg")


def remove_background(input_path: Path, output_path: Path) -> bool:
    """
    Remove background from a single image.
    
    Args:
        input_path: Path to input image
        output_path: Path to save output (PNG with alpha)
    
    Returns:
        True if successful, False otherwise
    """
    if not REMBG_AVAILABLE:
        return False
    
    try:
        from PIL import Image
        import cv2
        
        # Read image
        img = cv2.imread(str(input_path))
        if img is None:
            return False
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Remove background
        output = remove(img_rgb)
        
        # Convert back to BGRA (BGR with alpha)
        output_bgra = cv2.cvtColor(output, cv2.COLOR_RGBA2BGRA)
        
        # Save as PNG
        cv2.imwrite(str(output_path), output_bgra)
        
        return True
    except Exception as e:
        print(f"[ERROR] Failed to process {input_path}: {e}")
        return False


def process_folder(
    src_dir: Path,
    dst_dir: Path,
    extensions: list = None
) -> int:
    """
    Process all images in a folder and save with transparent backgrounds.
    
    Args:
        src_dir: Source directory with images
        dst_dir: Destination directory
        extensions: List of valid extensions
    
    Returns:
        Number of images processed
    """
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png"]
    
    if not REMBG_AVAILABLE:
        print("[ERROR] Rembg not available. Install: pip install rembg")
        return 0
    
    # Create output directory
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_files = []
    for ext in extensions:
        image_files.extend(src_dir.glob(f"*{ext}"))
        image_files.extend(src_dir.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"[WARNING] No images found in {src_dir}")
        return 0
    
    print(f"[INFO] Processing {len(image_files)} images...")
    
    # Process each image
    success_count = 0
    for img_path in tqdm(image_files, desc="Removing background"):
        # Output as PNG
        output_path = dst_dir / f"{img_path.stem}.png"
        
        if remove_background(img_path, output_path):
            success_count += 1
    
    print(f"[OK] Processed {success_count}/{len(image_files)} images")
    return success_count


def process_paired_dataset(
    src_dir: Path,
    dst_dir: Path,
    preserve_originals: bool = True
) -> int:
    """
    Process paired dataset (input/ and target/ folders).
    
    Args:
        src_dir: Source directory with input/ and target/ subfolders
        dst_dir: Destination directory
        preserve_originals: If True, keep original non-bg-removed images
    
    Returns:
        Total images processed
    """
    src_input = src_dir / "input"
    src_target = src_dir / "target"
    
    if not src_input.exists() or not src_target.exists():
        print(f"[ERROR] Expected input/ and target/ in {src_dir}")
        return 0
    
    # Create destination structure
    dst_input = dst_dir / "input"
    dst_target = dst_dir / "target"
    
    dst_input.mkdir(parents=True, exist_ok=True)
    dst_target.mkdir(parents=True, exist_ok=True)
    
    # Process input folder
    print("\n[INFO] Processing input folder...")
    input_count = process_folder(src_input, dst_input)
    
    # Process target folder
    print("\n[INFO] Processing target folder...")
    target_count = process_folder(src_target, dst_target)
    
    total = input_count + target_count
    print(f"\n[OK] Total processed: {total} images")
    
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Remove background from passport photos using Rembg"
    )
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Source dataset directory (contains input/ and target/)"
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="dataset_no_bg",
        help="Destination directory for background-removed images"
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install rembg dependency"
    )
    
    args = parser.parse_args()
    
    # Install rembg if requested
    if args.install:
        print("[INFO] Installing rembg...")
        import subprocess
        subprocess.run(["pip", "install", "rembg", "pillow", "opencv-python"], check=True)
        print("[OK] Rembg installed")
        return
    
    # Check rembg availability
    if not REMBG_AVAILABLE:
        print("[ERROR] Rembg not installed.")
        print("Install with: pip install rembg")
        print("Or run: python data/rm_background.py --src dataset --dst dataset_no_bg --install")
        return
    
    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    
    print(f"[INFO] Source: {src_dir}")
    print(f"[INFO] Destination: {dst_dir}")
    
    # Check if paired dataset
    if (src_dir / "input").exists() and (src_dir / "target").exists():
        process_paired_dataset(src_dir, dst_dir)
    else:
        # Single folder
        process_folder(src_dir, dst_dir)
    
    print(f"\n[OK] Background removal complete!")
    print(f"Output saved to: {dst_dir}")


if __name__ == "__main__":
    main()
