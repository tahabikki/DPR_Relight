#!/usr/bin/env python3
"""
Debug script to print all layer names in the model.
Run this once to understand the actual layer structure.
"""
import sys
from pathlib import Path
import torch

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "model"))

from defineHourglass_512_gray_skip import HourglassNet

# Create model
model = HourglassNet(baseFilter=16)

print("=" * 80)
print("ALL LAYER NAMES IN HOURGLASS MODEL")
print("=" * 80)

for i, (name, param) in enumerate(model.named_parameters()):
    print(f"{i:3d}. {name:60s} | Shape: {tuple(param.shape)}")

print("\n" + "=" * 80)
print("LAYER NAME PATTERNS (for freezing logic)")
print("=" * 80)

# Extract unique prefixes
prefixes = set()
for name, _ in model.named_parameters():
    prefix = name.split('.')[0]  # Get first part before the dot
    prefixes.add(prefix)

print("Unique layer prefixes:")
for prefix in sorted(prefixes):
    count = sum(1 for name, _ in model.named_parameters() if name.startswith(prefix))
    print(f"  - {prefix:15s} : {count:3d} parameters")

print("\n" + "=" * 80)
print("SUGGESTED FREEZING PATTERNS")
print("=" * 80)
print("""
For transfer learning / fine-tuning, you typically want to freeze:
  - pre_conv, pre_bn (initial preprocessing)
  - HG3, HG2, HG1 (early hourglass blocks - encoder)
  
And keep trainable:
  - HG0 (bottleneck - often modified)
  - conv_1, conv_2, conv_3 (post-processing)
  - output (new output layer)

This way you preserve pretrained features but adapt to your task.
""")

print("=" * 80)
