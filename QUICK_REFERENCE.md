# Quick Reference Card

## Installation (2 minutes)

```bash
# Create environment
conda env create -f environment.yml
conda activate gan-defect-augmentation

# Or with pip
pip install -r requirements.txt
```

## Dataset Setup (30 minutes)

```bash
# Download MVTec AD (~5.3 GB)
python download_mvtec.py

# Verify setup
python src/main.py --config config.yaml
```

## Configuration

Edit `config.yaml`:

```yaml
data:
  batch_size: 32          # Reduce if OOM
  num_workers: 8          # Increase for faster loading
  image_size: 256

gan:
  epochs: 200             # Reduce for testing
  learning_rate_g: 1e-4
  learning_rate_d: 2e-4

training:
  device: cuda
  mixed_precision: true   # Enable for speed
  wandb_project: gan-defect-augmentation
```

## Common Commands

### Data Loading
```python
from src.data.mvtec_dataset import MVTecDataLoader

loader = MVTecDataLoader(
    root_dir="./data/raw/mvtec",
    category="bottle",
    batch_size=32,
    num_workers=8,
)
train_loader, val_loader, test_loader = loader.get_loaders()
```

### Config Management
```python
from src.utils.config import load_config, create_directories

config = load_config("config.yaml")
create_directories(config)
```

### Logging
```python
from src.utils.logger import setup_logger

logger = setup_logger("my_module", config.training.log_dir)
logger.info("Training started")
```

## File Structure

```
gan-defect-augmentation/
├── config.yaml                  # Main config
├── environment.yml              # Conda env
├── download_mvtec.py            # Dataset download
├── src/
│   ├── main.py                  # Entry point
│   ├── data/
│   │   ├── mvtec_dataset.py     # Dataset class
│   │   └── transforms.py        # Augmentations
│   ├── models/                  # (Phase 2+)
│   └── utils/
│       ├── config.py
│       ├── logger.py
│       └── metrics.py
├── notebooks/
│   ├── 01_setup.ipynb           # Phase 1
│   ├── 02_gan_training.ipynb    # Phase 2
│   └── ...
├── data/raw/mvtec/              # Dataset
├── checkpoints/                 # Models
├── logs/                        # Logs
└── outputs/                     # Results
```

## MVTec AD Categories

```
bottle, cable, capsule, carpet, grid,
hazelnut, leather, metal_nut, pill, screw,
tile, toothbrush, transistor, wood, zipper
```

## Troubleshooting

### CUDA Out of Memory
```yaml
# config.yaml
data:
  batch_size: 16
  num_workers: 4
training:
  mixed_precision: true
```

### Slow Data Loading
```yaml
# config.yaml
data:
  num_workers: 8  # Increase
  batch_size: 32  # Increase
```

### Dataset Not Found
```bash
# Check structure
ls -la data/raw/mvtec/bottle/train/good/

# Re-download
rm -rf data/raw/mvtec/
python download_mvtec.py
```

## Performance Tips

1. **GPU Memory**: Enable mixed precision
2. **Data Loading**: Increase num_workers
3. **Training Speed**: Increase batch_size
4. **Reproducibility**: Set seed in config

## Monitoring

### wandb
```bash
wandb login
# View at: https://wandb.ai/your-username/gan-defect-augmentation
```

### TensorBoard
```bash
tensorboard --logdir logs/
# Open http://localhost:6006
```

### GPU Usage
```bash
nvidia-smi
watch -n 1 nvidia-smi  # Update every 1 second
```

## Dataset Info

| Aspect | Value |
|--------|-------|
| Categories | 15 |
| Total Images | ~15,000 |
| Per Category | ~1,000 |
| Image Size | 256×256 |
| Train/Val/Test | 80/10/10 |
| Defect Types | 3-5 per category |

## Expected Results

| Phase | Output | Time |
|-------|--------|------|
| 1 | Dataset ready | 30 min |
| 2 | 1000+ synthetic/cat | 2-3 days |
| 3 | 500+ filtered/cat | 2 hours |
| 4 | 3 trained models | 4 hours |
| 5 | Visualizations | 30 min |
| 6 | API deployed | 1 hour |

## Key Metrics

### GAN Training
- FID Score: < 20 (lower is better)
- Training Time: 2-3 days
- Convergence: ~epoch 150-200

### Classifier
- Baseline F1: ~85%
- Traditional F1: ~88%
- GAN-Aug F1: ~95% (+10-20%)

## Hyperparameters

```yaml
# Data
image_size: 256
batch_size: 32
num_workers: 8

# GAN
latent_dim: 128
lr_g: 1e-4
lr_d: 2e-4
gp_weight: 10.0
d_steps: 5

# Classifier
model: efficientnet_b2
epochs: 100
lr: 1e-3
label_smoothing: 0.1
```

## Phases Overview

```
Phase 1: Setup ✅
  └─ Download dataset, setup pipeline

Phase 2: GAN Training 🔄
  └─ Train WGAN-GP, generate synthetics

Phase 3: Quality Control 🔄
  └─ Filter by FID, LPIPS, coverage

Phase 4: Classifier 🔄
  └─ Train 3 models, compare performance

Phase 5: Visualization 🔄
  └─ Generate plots, dashboard, report

Phase 6: Deployment 🔄
  └─ FastAPI, Docker, Kubernetes
```

## Next Steps

1. **Now**: Phase 1 complete ✅
2. **Next**: `python src/train_gan.py --config config.yaml`
3. **Then**: `python src/evaluate_quality.py --config config.yaml`
4. **Then**: `python src/train_classifier.py --config config.yaml`
5. **Then**: `jupyter notebook notebooks/05_visualization.ipynb`
6. **Finally**: `docker build -t gan-defect . && docker run -p 8000:8000 gan-defect`

## Documentation

- `README.md` - Full documentation
- `QUICKSTART.md` - 5-minute guide
- `PROJECT_OVERVIEW.md` - Architecture
- `IMPLEMENTATION_CHECKLIST.md` - Detailed checklist
- `PHASE1_COMPLETE.md` - Phase 1 summary

## Support

- **Issues**: GitHub Issues
- **Questions**: See README.md
- **Bugs**: Create issue with logs

---

**Status**: Phase 1 Complete ✅ | Ready for Phase 2 →

**Time to Phase 2**: ~30 minutes (after dataset download)
