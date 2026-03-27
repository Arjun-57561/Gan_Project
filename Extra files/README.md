# GAN Defect Augmentation for MVTec AD

A production-grade implementation of a Defect Transfer GAN (DT-GAN) for synthetic defect generation and data augmentation on the MVTec AD dataset.

## Project Overview

This project implements a complete pipeline for:
1. **GAN Training**: WGAN-GP with defect-background disentanglement
2. **Synthetic Image Generation**: 1000+ high-quality defect images per category
3. **Quality Control**: Multi-metric filtering to keep only top-quality synthetics
4. **Classifier Training**: EfficientNet with augmentation comparison
5. **Experiment Tracking**: Comprehensive wandb logging and visualization
6. **Production Deployment**: FastAPI endpoints with Docker containerization

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate gan-defect-augmentation

# Or install with pip
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
# Download and prepare MVTec AD dataset (~5.3 GB)
python download_mvtec.py
```

This will:
- Download all 15 MVTec AD categories
- Extract to `./data/raw/mvtec/`
- Create 80/10/10 train/val/test splits
- Verify dataset structure

### 3. Test Data Loading

```bash
# Run setup notebook to verify everything works
jupyter notebook notebooks/01_setup.ipynb

# Or test from command line
python src/main.py --config config.yaml
```

## Project Structure

```
gan-defect-augmentation/
├── environment.yml              # Conda environment
├── config.yaml                  # Main configuration
├── requirements.txt             # Pip requirements
├── README.md                    # This file
├── download_mvtec.py            # Dataset download script
├── src/
│   ├── __init__.py
│   ├── main.py                  # Entry point
│   ├── train_gan.py             # GAN training (Phase 2)
│   ├── evaluate_quality.py      # Quality filtering (Phase 3)
│   ├── train_classifier.py      # Classifier training (Phase 4)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── mvtec_dataset.py     # MVTec dataset class
│   │   └── transforms.py        # Data augmentation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── generator.py         # Generator architecture
│   │   └── discriminator.py     # Discriminator architecture
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # Config management
│       ├── logger.py            # Logging setup
│       └── metrics.py           # Metrics computation
├── notebooks/
│   ├── 01_setup.ipynb           # Phase 1: Setup
│   ├── 02_gan_training.ipynb    # Phase 2: GAN training
│   ├── 03_quality_control.ipynb # Phase 3: Quality filtering
│   ├── 04_classifier.ipynb      # Phase 4: Classifier training
│   └── 05_visualization.ipynb   # Phase 5: Results visualization
├── data/
│   ├── raw/
│   │   └── mvtec/               # Downloaded dataset
│   └── processed/               # Processed data
├── checkpoints/                 # Model checkpoints
├── logs/                        # Training logs
└── outputs/                     # Generated outputs
```

## Configuration

Edit `config.yaml` to customize:

```yaml
data:
  image_size: 256
  batch_size: 32
  num_workers: 8

gan:
  learning_rate_g: 1e-4
  learning_rate_d: 2e-4
  epochs: 200
  gradient_penalty_weight: 10.0

classifier:
  model_name: efficientnet_b2
  epochs: 100
  label_smoothing: 0.1

training:
  device: cuda
  mixed_precision: true
  wandb_project: gan-defect-augmentation
```

## Phase-by-Phase Implementation

### Phase 1: Environment & Dataset Setup ✅
- [x] Conda environment with all dependencies
- [x] MVTec AD dataset download and preparation
- [x] Data loading pipeline with transforms
- [x] Configuration management
- [x] Setup verification notebook

**Status**: Complete and tested

### Phase 2: GAN Architecture & Training
- [ ] Generator (U-Net with conditional instance norm)
- [ ] Discriminator (PatchGAN multi-scale)
- [ ] WGAN-GP training loop
- [ ] Checkpoint management
- [ ] wandb logging

**Next**: Run `python src/train_gan.py --config config.yaml`

### Phase 3: Synthetic Image Quality Control
- [ ] Multi-metric scoring (FID, LPIPS, coverage, sharpness)
- [ ] DINOv2 feature extraction
- [ ] Nearest neighbor search
- [ ] Filtering pipeline
- [ ] Quality visualization dashboard

### Phase 4: Downstream Classifier & Comparison
- [ ] EfficientNet-B2 classifier
- [ ] Three training regimes (baseline, traditional, GAN-aug)
- [ ] Focal loss and label smoothing
- [ ] Test-time augmentation
- [ ] Ablation studies

### Phase 5: Experiment Tracking & Visualization
- [ ] wandb project setup
- [ ] Training curves and metrics
- [ ] Generated image samples
- [ ] t-SNE visualization
- [ ] Interactive Plotly dashboard
- [ ] PDF report generation

### Phase 6: Production Deployment
- [ ] FastAPI endpoints
- [ ] Docker containerization
- [ ] Kubernetes deployment
- [ ] Model serving with TorchServe

## MVTec AD Categories

The dataset includes 15 object categories:
- bottle, cable, capsule, carpet, grid
- hazelnut, leather, metal_nut, pill, screw
- tile, toothbrush, transistor, wood, zipper

Each category has:
- Normal images (good)
- Defective images with multiple defect types
- Binary ground truth masks for defect regions

## Key Features

### Data Pipeline
- ✅ Custom PyTorch Dataset class
- ✅ Albumentations transforms (resize, flip, rotate, brightness/contrast)
- ✅ Binary defect masks (1=defect, 0=normal)
- ✅ Efficient data loading with multiprocessing
- ✅ Train/val/test splits (80/10/10)

### GAN Architecture
- WGAN-GP loss for stable training
- Spectral normalization on all layers
- Conditional instance normalization
- U-Net generator with skip connections
- Multi-scale PatchGAN discriminator

### Quality Control
- FID score computation
- LPIPS perceptual distance
- Defect coverage analysis
- Sharpness metrics
- DINOv2 feature similarity

### Classifier Training
- EfficientNet-B2 backbone
- Focal loss for class imbalance
- Label smoothing (0.1)
- Test-time augmentation
- Hyperparameter sweeps with wandb

## Dependencies

- PyTorch 2.1+ with CUDA 12.1
- torchvision, torchaudio
- albumentations (data augmentation)
- pytorch-fid (FID evaluation)
- timm (EfficientNet)
- wandb (experiment tracking)
- omegaconf (configuration)
- scikit-learn, pandas, matplotlib, seaborn, plotly

## Hardware Requirements

- **Minimum**: GPU with 8GB VRAM (RTX 3060)
- **Recommended**: GPU with 24GB+ VRAM (RTX 4090, A100)
- **Training time**: 2-3 days for full pipeline on RTX 4090

## Usage Examples

### Test Data Loading
```python
from src.data.mvtec_dataset import MVTecDataLoader

loader = MVTecDataLoader(
    root_dir="./data/raw/mvtec",
    category="bottle",
    batch_size=32,
    num_workers=8,
)
train_loader, val_loader, test_loader = loader.get_loaders()

batch = next(iter(train_loader))
print(batch['image'].shape)  # (32, 3, 256, 256)
print(batch['mask'].shape)   # (32, 256, 256)
print(batch['label'].shape)  # (32,)
```

### Load Configuration
```python
from src.utils.config import load_config, create_directories

config = load_config("config.yaml")
create_directories(config)

print(config.data.batch_size)  # 32
print(config.gan.epochs)       # 200
```

## Experiment Tracking

All experiments are logged to wandb:

```bash
# Set wandb project
export WANDB_PROJECT=gan-defect-augmentation

# Login to wandb
wandb login

# Training automatically logs to wandb
python src/train_gan.py --config config.yaml
```

View results at: https://wandb.ai/your-username/gan-defect-augmentation

## Expected Results

### GAN Training
- FID score: < 20 (vs real defects)
- Training time: 2-3 days on RTX 4090
- Generated images: 1000+ per category

### Classifier Performance
- Baseline (real only): ~85% F1-macro
- Traditional augmentation: ~88% F1-macro
- GAN augmentation: ~95% F1-macro (+10-20% improvement)

## Troubleshooting

### CUDA Out of Memory
- Reduce batch_size in config.yaml
- Enable gradient_accumulation_steps
- Use mixed_precision: true

### Dataset Download Issues
- Check internet connection
- Verify disk space (>10GB needed)
- Try manual download from: https://www.mvtec.com/company/research/datasets/mvtec-ad

### Data Loading Errors
- Verify dataset structure: `data/raw/mvtec/[category]/[train|val|test]/`
- Check image formats (should be .png)
- Ensure masks exist in `ground_truth/` directory

## References

- MVTec AD Dataset: https://www.mvtec.com/company/research/datasets/mvtec-ad
- WGAN-GP: https://arxiv.org/abs/1704.00028
- DT-GAN: https://arxiv.org/abs/2203.08270
- EfficientNet: https://arxiv.org/abs/1905.11946
- DINOv2: https://arxiv.org/abs/2304.07193

## License

This project is provided as-is for research and educational purposes.

## Citation

If you use this code, please cite:

```bibtex
@software{gan-defect-augmentation,
  title={GAN Defect Augmentation for MVTec AD},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## Support

For issues, questions, or contributions:
1. Check existing issues on GitHub
2. Create a new issue with detailed description
3. Include error logs and configuration
4. Provide minimal reproducible example

## Roadmap

- [ ] Phase 2: GAN training implementation
- [ ] Phase 3: Quality control pipeline
- [ ] Phase 4: Classifier training and evaluation
- [ ] Phase 5: Visualization dashboard
- [ ] Phase 6: Production deployment
- [ ] Multi-GPU training support
- [ ] Distributed training with DDP
- [ ] Model quantization for inference
- [ ] ONNX export for deployment
