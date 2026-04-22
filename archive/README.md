# Archive

This folder contains files from the original DPR repository that are preserved for reference but not essential for the fine-tuning pipeline.

## Contents

### `README_Finetune.md`
The original fine-tuning requirements document. Contains the complete specification and task breakdown for this project.

### `demos/`
- `testNetwork_demo_512.py` — Inference demo on 512×512 pretrained model. Shows how to load checkpoints and process images through the original DPR pipeline.
- `testNetwork_demo_1024.py` — Inference demo on 1024×1024 pretrained model. Demonstrates multi-scale fine-tuning approach.

These are kept for reference only. For production inference, use `scripts/infer.py` instead.

### `result/`
Old demonstration outputs from the original DPR demos (example relighting results on `data/obama.jpg`). These are replaced by training artifacts in `checkpoints/` once fine-tuning begins.

## Why Archived?

The original DPR project includes many reference files and demos tied to the synthetic data generation pipeline (`zhhoper/RI_render_DPR`). Since this fine-tuning project:
- Uses **real paired passport photos** instead of synthetic data
- Focuses on **neutral lighting** instead of arbitrary relighting
- Employs a **simplified training pipeline** (no rendering, direct image pairs)

...these reference materials became redundant. However, they're preserved here to:
1. Maintain provenance with the original DPR codebase
2. Provide reference implementations if debugging or extending is needed
3. Document the evolution from paper→original implementation→fine-tuning variant

## When to Use Files in Archive

**Use `demos/` if you want to:**
- Understand the original DPR architecture in action
- Test the pretrained checkpoints on arbitrary lighting vectors
- Refer to the original inference pipeline (though `scripts/infer.py` is now the recommended entry point)

**Everything else** should come from the top-level `scripts/`, `configs/`, and `data/` directories for the fine-tuning workflow.
