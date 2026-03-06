# Quick Start Guide

Get the GAN defect augmentation project running in 5 minutes.

## Prerequisites

- Python 3.11+
- CUDA 12.1 (for GPU support)
- 10GB+ disk space for dataset
- GPU with 8GB+ VRAM (recommended)

## Installation

### Option 1: Conda (Recommended)

```bash
# Create environment
conda env create -f environment.yml

# Activate
conda activate gan-defect-augmentation
```

### Option 2: Pip

```bash
pip install -r requirements.txt
```

## Setup (5 minutes)

### 1. Download Dataset

```bash
python download_mvtec.py
```

**What it does:**
- Downloads MVTec AD (~5.3 GB)
- Extracts to `./data/raw/mvtec/`
- Creates train/val/test splits
- Verifies structure

**Time**: ~30 minutes (depends on internet speed)

### 2. Verify Installation

```bash
python src/main.py --config config.yaml
```

**Expected output:**
```
Loading bottle...
Train batches: 42
Val batches: 5
Test batches: 5
Batch shapes - Image: torch.Size([32, 3, 256, 256]), Mask: torch.Size([32, 256, 256])
Setup completed successfully
```

### 3. Test Jupyter Notebook

```bash
jupyter notebook notebooks/01_setup.ipynb
```

Run all cells to verify data loading and visualization.

## Next Steps

### Phase 2: Train GAN

```bash
python src/train_gan.py --config config.yaml
```

**Expected:**
- Training time: 2-3 days on RTX 4090
- Generates 1000+ synthetic defects per category
- Logs to wandb automatically

### Phase 3: Filter Synthetic Images

```bash
python src/evaluate_quality.py --config config.yaml
```

**Expected:**
- Keeps top 50% highest quality images
- Computes FID, LPIPS, coverage scores
- Generates quality report

### Phase 4: Train Classifier

```bash
python src/train_classifier.py --config config.yaml
```

**Expected:**
- Trains 3 models (baseline, traditional, GAN-aug)
- Compares performance
- Generates comparison tables

## Configuration

Edit `config.yaml` for quick adjustments:

```yaml
# Reduce batch size if out of memory
data:
  batch_size: 16  # Default: 32

# Reduce epochs for quick testing
gan:
  epochs: 10  # Default: 200

# Disable wandb logging
training:
  wandb_project: null
```

## Common Issues

### CUDA Out of Memory

```yaml
# In config.yaml
data:
  batch_size: 16
  num_workers: 4

training:
  mixed_precision: true
```

### Dataset Not Found

```bash
# Verify dataset structure
ls -la data/raw/mvtec/bottle/train/good/
# Should show .png files

# Re-download if needed
rm -rf data/raw/mvtec/
python download_mvtec.py
```

### Slow Data Loading

```yaml
# In config.yaml
data:
  num_workers: 8  # Increase if you have CPU cores
  batch_size: 32  # Increase for better throughput
```

## File Structure After Setup

```
gan-defect-augmentation/
├── data/
│   └── raw/
│       └── mvtec/
│           ├── bottle/
│           │   ├── train/
│           │   ├── val/
│           │   └── test/
│           ├── cable/
│           └── ... (13 more categories)
├── checkpoints/
├── logs/
├── outputs/
└── src/
```

## Monitoring Training

### With wandb

```bash
# Login first
wandb login

# View dashboard
# https://wandb.ai/your-username/gan-defect-augmentation
```

### With TensorBoard

```bash
tensorboard --logdir logs/
# Open http://localhost:6006
```

## Performance Benchmarks

| Phase | Time | GPU Memory | Output |
|-------|------|-----------|--------|
| Data Setup | 30 min | - | 15 categories |
| GAN Training | 2-3 days | 24GB | 15k synthetic images |
| Quality Filter | 2 hours | 8GB | 7.5k filtered images |
| Classifier Train | 4 hours | 12GB | 3 models + metrics |
| Visualization | 30 min | 4GB | Dashboard + report |

## Useful Commands

```bash
# Check GPU usage
nvidia-smi

# Monitor training
tail -f logs/gan-defect.log

# Count images
find data/raw/mvtec -name "*.png" | wc -l

# Clean outputs
rm -rf checkpoints/* logs/* outputs/*

# Reset everything
rm -rf data/ checkpoints/ logs/ outputs/
```

## Next: Phase 2

Ready to train the GAN? See `README.md` for detailed Phase 2 instructions.

```bash
python src/train_gan.py --config config.yaml
```

## Support

- **Issues**: Check GitHub issues
- **Questions**: See README.md FAQ section
- **Bugs**: Create issue with error log and config

## What's Next?

After Phase 1 setup:
1. ✅ Phase 1: Environment & Dataset (YOU ARE HERE)
2. → Phase 2: GAN Training
3. → Phase 3: Quality Control
4. → Phase 4: Classifier Training
5. → Phase 5: Visualization
6. → Phase 6: Deployment

Each phase builds on the previous one. Start with Phase 2 when ready!
