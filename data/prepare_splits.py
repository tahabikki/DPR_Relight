#!/usr/bin/env python3
"""
Dataset Splitting Script for DPR Passport Photo Fine-tuning

This script reads paired image data from input/ and target/ folders,
splits it into train/eval/test sets, and creates .lst files compatible
with DPR's training pipeline.

Usage:
    python data/prepare_splits.py --src dataset --dst dataset_split --seed 42

The script:
- Preserves pairing: same basename always goes to the same split
- Creates symlinks (or copies) to avoid duplicating large image files
- Generates .lst files (one pair per line: input_path<TAB>target_path)
- Detects and reports orphan files (unmatched input/target pairs)
- Uses pathlib.Path for Windows compatibility
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import random


def find_paired_files(input_dir: Path, target_dir: Path) -> Tuple[List[str], List[str]]:
    """
    Find all paired input/target files.
    
    Args:
        input_dir: Path to input/ folder
        target_dir: Path to target/ folder
    
    Returns:
        (paired_basenames, orphan_input, orphan_target)
    """
    input_files = set(f.stem for f in input_dir.glob("*.[jJ][pP][gG]"))
    target_files = set(f.stem for f in target_dir.glob("*.[jJ][pP][gG]"))
    
    paired = sorted(list(input_files & target_files))
    orphan_input = input_files - target_files
    orphan_target = target_files - input_files
    
    return paired, orphan_input, orphan_target


def split_dataset(
    paired_files: List[str],
    train_ratio: float = 0.7,
    eval_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    Split paired files into train/eval/test sets.
    
    Args:
        paired_files: List of basenames (without extension)
        train_ratio, eval_ratio, test_ratio: Split ratios (should sum to ~1.0)
        seed: Random seed for reproducibility
    
    Returns:
        Dict with keys 'train', 'eval', 'test' mapping to lists of basenames
    """
    random.seed(seed)
    files_shuffled = paired_files.copy()
    random.shuffle(files_shuffled)
    
    n_total = len(files_shuffled)
    n_train = int(n_total * train_ratio)
    n_eval = int(n_total * eval_ratio)
    
    splits = {
        'train': files_shuffled[:n_train],
        'eval': files_shuffled[n_train:n_train + n_eval],
        'test': files_shuffled[n_train + n_eval:]
    }
    
    return splits


def create_split_structure(
    src_dir: Path,
    dst_dir: Path,
    splits: Dict[str, List[str]],
    use_symlink: bool = False
) -> Dict[str, int]:
    """
    Create the dataset_split/ directory structure with input/ and target/ subdirs.
    
    Args:
        src_dir: Source dataset root (contains input/ and target/)
        dst_dir: Destination dataset_split/ root
        splits: Dict of split names to list of basenames
        use_symlink: If True, create symlinks; if False, copy files
    
    Returns:
        Dict with counts per split
    """
    input_src = src_dir / "input"
    target_src = src_dir / "target"
    
    counts = {}
    
    for split_name, basenames in splits.items():
        split_input_dir = dst_dir / split_name / "input"
        split_target_dir = dst_dir / split_name / "target"
        
        split_input_dir.mkdir(parents=True, exist_ok=True)
        split_target_dir.mkdir(parents=True, exist_ok=True)
        
        for basename in basenames:
            # Find the actual file (could be .jpg or .jpeg)
            input_files = list(input_src.glob(f"{basename}.*"))
            target_files = list(target_src.glob(f"{basename}.*"))
            
            if input_files and target_files:
                input_src_file = input_files[0]
                target_src_file = target_files[0]
                
                input_dst_file = split_input_dir / input_src_file.name
                target_dst_file = split_target_dir / target_src_file.name
                
                if use_symlink:
                    # Create symlinks to save space
                    if input_dst_file.exists():
                        input_dst_file.unlink()
                    if target_dst_file.exists():
                        target_dst_file.unlink()
                    
                    input_dst_file.symlink_to(input_src_file.resolve())
                    target_dst_file.symlink_to(target_src_file.resolve())
                else:
                    # Copy files (safer on Windows, uses more space)
                    shutil.copy2(input_src_file, input_dst_file)
                    shutil.copy2(target_src_file, target_dst_file)
        
        counts[split_name] = len(basenames)
    
    return counts


def create_lst_files(
    src_dir: Path,
    dst_dir: Path,
    splits: Dict[str, List[str]]
) -> None:
    """
    Create .lst files for DPR compatibility.
    
    Each line contains: input_rel_path<TAB>target_rel_path
    where paths are relative to dst_dir.
    """
    input_src = src_dir / "input"
    target_src = src_dir / "target"
    
    for split_name, basenames in splits.items():
        lst_file = dst_dir / f"{split_name}.lst"
        
        with open(lst_file, 'w') as f:
            for basename in basenames:
                # Find actual filenames
                input_files = list(input_src.glob(f"{basename}.*"))
                target_files = list(target_src.glob(f"{basename}.*"))
                
                if input_files and target_files:
                    input_name = input_files[0].name
                    target_name = target_files[0].name
                    
                    input_rel_path = f"{split_name}/input/{input_name}"
                    target_rel_path = f"{split_name}/target/{target_name}"
                    
                    f.write(f"{input_rel_path}\t{target_rel_path}\n")
        
        print(f"✓ Created: {lst_file}")


def print_summary(
    paired_count: int,
    orphan_input: set,
    orphan_target: set,
    counts: Dict[str, int]
) -> None:
    """Print a summary of the split operation."""
    print("\n" + "=" * 70)
    print("DATASET SPLIT SUMMARY")
    print("=" * 70)
    print(f"\nTotal paired images: {paired_count}")
    print(f"  Train:  {counts['train']} ({counts['train']/paired_count*100:.1f}%)")
    print(f"  Eval:   {counts['eval']} ({counts['eval']/paired_count*100:.1f}%)")
    print(f"  Test:   {counts['test']} ({counts['test']/paired_count*100:.1f}%)")
    
    if orphan_input or orphan_target:
        print("\n⚠ WARNING: Orphan files detected (unpaired):")
        if orphan_input:
            print(f"\n  Input files without target ({len(orphan_input)}):")
            for name in sorted(orphan_input)[:5]:
                print(f"    - {name}")
            if len(orphan_input) > 5:
                print(f"    ... and {len(orphan_input) - 5} more")
        
        if orphan_target:
            print(f"\n  Target files without input ({len(orphan_target)}):")
            for name in sorted(orphan_target)[:5]:
                print(f"    - {name}")
            if len(orphan_target) > 5:
                print(f"    ... and {len(orphan_target) - 5} more")
    else:
        print("\n✓ No orphan files - all pairs matched!")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Split DPR dataset into train/eval/test sets"
    )
    parser.add_argument(
        "--src",
        type=str,
        default="dataset",
        help="Source dataset directory (contains input/ and target/ subdirs)"
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="dataset_split",
        help="Destination root for split dataset"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of data for training (default: 0.7)"
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.2,
        help="Fraction of data for evaluation (default: 0.2)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of data for testing (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)"
    )
    parser.add_argument(
        "--use-symlink",
        action="store_true",
        help="Use symlinks instead of copying files (saves space, may not work on all systems)"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    src_dir = Path(args.src).resolve()
    dst_dir = Path(args.dst).resolve()
    
    # Validate source directory
    if not src_dir.exists():
        print(f"❌ Error: Source directory not found: {src_dir}")
        return False
    
    input_dir = src_dir / "input"
    target_dir = src_dir / "target"
    
    if not input_dir.exists() or not target_dir.exists():
        print(f"❌ Error: Expected input/ and target/ subdirs in {src_dir}")
        return False
    
    print(f"\n📂 Source directory: {src_dir}")
    print(f"📂 Destination directory: {dst_dir}")
    print(f"🌱 Random seed: {args.seed}")
    
    # Find paired files
    print("\n🔍 Scanning for paired files...")
    paired, orphan_input, orphan_target = find_paired_files(input_dir, target_dir)
    
    if not paired:
        print("❌ Error: No paired files found!")
        return False
    
    print(f"✓ Found {len(paired)} paired images")
    
    # Split dataset
    print("\n✂️ Splitting dataset...")
    splits = split_dataset(
        paired,
        train_ratio=args.train_ratio,
        eval_ratio=args.eval_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Create directory structure
    print("\n📁 Creating directory structure...")
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    counts = create_split_structure(
        src_dir,
        dst_dir,
        splits,
        use_symlink=args.use_symlink
    )
    print("✓ Directory structure created")
    
    # Create .lst files
    print("\n📝 Creating .lst files...")
    create_lst_files(src_dir, dst_dir, splits)
    
    # Print summary
    print_summary(len(paired), orphan_input, orphan_target, counts)
    
    print(f"\n✅ Dataset split complete!")
    print(f"   Split dataset available at: {dst_dir}")
    print(f"   Use: python scripts/train.py --config configs/finetune_passport.yaml")
    
    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
