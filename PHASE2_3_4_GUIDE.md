# Phases 2-4: GAN Training, Quality Control & Classifier

Complete guide for implementing and running Phases 2, 3, and 4.

## Phase 2: GAN Architecture & Training

### What's Implemented

#### Generator Architecture
- **Type**: U-Net with conditional instance normalization
- **Input**: (B, 6, 256, 256) = [normal_image (3ch) + defect_mask (1ch) + padding (2ch)]
- **Output**: (B, 3, 256, 256) = synthetic defective image
- **Key Features**:
  - Spectral normalization on all conv layers
  - Conditional instance norm (conditioned on defect type)
  - Skip connections between encoder/decoder
  - Residual blocks for stable training
  - Tanh activation on output

#### Discriminator Architecture
- **Type**: Multi-scale PatchGAN
- **Input**: (B, 4, 256, 256) = [image (3ch) + defect_mask (1ch)]
- **Output**: (B, 1, 1, 1) = validity score
- **Key Features**:
  - 3 scales: full (256×256), half (128×128), quarter (64×64)
  - Spectral normalization
  - LeakyReLU(0.2) activation
  - Multi-scale validity scores averaged

#### Training Loop
- **Loss**: WGAN-GP (Wasserstein GAN with Gradient Penalty)
- **Gradient Penalty Weight**: 10.0
- **Discriminator Steps**: 5 per generator step
- **Learning Rates**: D=2e-4, G=1e-4
- **Optimizer**: Adam with betas=(0, 0.9)
- **Mixed Precision**: Enabled for speed
- **Gradient Accumulation**: Effective batch size = 128

### Files Created

```
src/models/
├── __init__.py
├── generator.py          # Generator with conditional instance norm
└── discriminator.py      # Multi-scale PatchGAN

src/train_gan.py          # Complete training script
notebooks/02_gan_training.ipynb  # Training notebook
```

### How to Run

#### Quick Test (Notebook)
```bash
jupyter notebook notebooks/02_gan_training.ipynb
```

This will:
- Initialize models
- Load data
- Test forward pass
- Visualize generated images

#### Full Training (Command Line)
```bash
python src/train_gan.py --config config.yaml
```

**Expected Output**:
- Training time: 2-3 days on RTX 4090
- FID score: < 20 (vs real defects)
- Generated images: 1000+ per category
- Checkpoints saved every 10 epochs

### Configuration

Edit `config.yaml`:

```yaml
gan:
  latent_dim: 128
  defect_embedding_dim: 64
  learning_rate_g: 1e-4
  learning_rate_d: 2e-4
  beta1: 0.0
  beta2: 0.9
  gradient_penalty_weight: 10.0
  discriminator_steps: 5
  epochs: 200
  early_stopping_patience: 20
  checkpoint_interval: 10
  log_interval: 100
  mixed_precision: true
  gradient_accumulation_steps: 4
```

### Monitoring Training

#### wandb Dashboard
```bash
# Login
wandb login

# View at: https://wandb.ai/your-username/gan-defect-augmentation
```

Logged metrics:
- Loss D, Loss G
- Gradient penalty
- Generated sample images (8×8 grid)
- FID score progression

#### TensorBoard
```bash
tensorboard --logdir logs/
# Open http://localhost:6006
```

#### GPU Monitoring
```bash
nvidia-smi
watch -n 1 nvidia-smi  # Update every 1 second
```

### Troubleshooting

#### CUDA Out of Memory
```yaml
# config.yaml
data:
  batch_size: 16  # Reduce from 32
  num_workers: 4  # Reduce from 8

training:
  mixed_precision: true  # Enable
```

#### Training Unstable
- Reduce learning rate
- Increase gradient penalty weight
- Increase discriminator steps

#### Slow Training
- Increase batch_size
- Increase num_workers
- Enable mixed_precision

---

## Phase 3: Synthetic Image Quality Control

### What's Implemented

#### Quality Metrics

1. **FID Score** (Fréchet Inception Distance)
   - Measures distribution similarity
   - Lower is better (target: < 25)
   - Computed on feature space

2. **LPIPS Distance** (Learned Perceptual Image Patch Similarity)
   - Perceptual distance to nearest real image
   - Lower is better (target: < 0.5)
   - Uses DINOv2 features

3. **Defect Coverage**
   - Overlap between synthetic and real defect masks
   - Higher is better (target: > 0.3)
   - Ensures synthetic defect is in right location

4. **Sharpness**
   - Laplacian variance
   - Detects blur
   - Higher is better (target: > 100)

#### Filtering Pipeline

```python
# Compute quality scores for all synthetic images
# Normalize each metric to [0, 1]
# Compute weighted average:
#   final_score = 0.3*FID + 0.3*LPIPS + 0.2*Coverage + 0.2*Sharpness
# Sort by final_score
# Keep top 50% (configurable)
```

### Files Created

```
src/evaluate_quality.py  # Quality evaluation pipeline
notebooks/03_quality_control.ipynb  # Quality analysis notebook
```

### How to Run

#### Quick Test (Notebook)
```bash
jupyter notebook notebooks/03_quality_control.ipynb
```

#### Full Evaluation (Command Line)
```bash
python src/evaluate_quality.py --config config.yaml
```

**Expected Output**:
- Quality scores CSV
- Filtered images directory
- Quality distribution plots
- Quality report

### Configuration

Edit `config.yaml`:

```yaml
quality_control:
  fid_threshold: 25.0
  lpips_threshold: 0.5
  defect_coverage_threshold: 0.3
  sharpness_threshold: 100.0
  keep_ratio: 0.5  # Keep top 50%
  dino_model: dinov2_vitb14
  feature_similarity_threshold: 0.7
```

### Output

After filtering:
- **Filtered Images**: `outputs/filtered_synthetic/`
- **Quality Scores**: `outputs/quality_scores.csv`
- **Distribution Plot**: `outputs/quality_distribution.png`
- **Quality Report**: `outputs/quality_report.txt`

### Expected Results

| Metric | Before Filter | After Filter |
|--------|---------------|--------------|
| Avg FID | 28 | 18 |
| Avg LPIPS | 0.65 | 0.35 |
| Avg Coverage | 0.25 | 0.45 |
| Images Kept | 100% | 50% |

---

## Phase 4: Downstream Classifier & Comparison

### What's Implemented

#### Three Training Regimes

1. **BASELINE**
   - Data: Real images only
   - Augmentation: Basic (resize, normalize)
   - Expected F1: ~85%

2. **TRADITIONAL**
   - Data: Real images only
   - Augmentation: Heavy (mosaic, mixup, cutmix)
   - Expected F1: ~88%

3. **GAN_AUG**
   - Data: Real + filtered synthetic (1:3 ratio)
   - Augmentation: Basic
   - Expected F1: ~95% (+10-20% improvement)

#### Classifier Architecture
- **Model**: EfficientNet-B2 (from timm)
- **Input**: (B, 3, 256, 256)
- **Output**: (B, 15) = logits for 15 categories
- **Pretrained**: ImageNet weights

#### Training Configuration
- **Loss**: Focal Loss (α=0.25, γ=2.0)
- **Label Smoothing**: 0.1
- **Optimizer**: AdamW
- **Learning Rate**: 1e-3
- **Scheduler**: Cosine annealing
- **Epochs**: 100
- **Warmup**: 5 epochs
- **Test-Time Augmentation**: 4 variants

### Files Created

```
src/train_classifier.py  # Classifier training script
notebooks/04_classifier.ipynb  # Classifier analysis notebook
```

### How to Run

#### Quick Test (Notebook)
```bash
jupyter notebook notebooks/04_classifier.ipynb
```

#### Full Training (Command Line)
```bash
python src/train_classifier.py --config config.yaml
```

**Expected Output**:
- Training time: 4 hours per model
- 3 trained models (baseline, traditional, GAN-aug)
- Performance comparison table
- Confusion matrices

### Configuration

Edit `config.yaml`:

```yaml
classifier:
  model_name: efficientnet_b2
  num_classes: 15
  learning_rate: 1e-3
  epochs: 100
  batch_size: 32
  label_smoothing: 0.1
  focal_loss_alpha: 0.25
  focal_loss_gamma: 2.0
  warmup_epochs: 5
  tta_enabled: true
  tta_augmentations: 4
```

### Evaluation Metrics

- **Accuracy**: Overall correctness
- **F1-Macro**: Unweighted average F1 (good for imbalanced data)
- **F1-Weighted**: Weighted by class frequency
- **F1-Rare**: F1 for rare defect types
- **AUC-ROC**: Area under ROC curve
- **Confusion Matrix**: Per-class breakdown

### Expected Results

| Method | Accuracy | F1-Macro | F1-Rare | AUC-ROC |
|--------|----------|----------|---------|---------|
| Baseline | 85% | 0.84 | 0.72 | 0.92 |
| Traditional | 88% | 0.87 | 0.78 | 0.94 |
| GAN-Aug | 95% | 0.94 | 0.89 | 0.97 |
| **Improvement** | **+10%** | **+10%** | **+17%** | **+5%** |

### Ablation Studies

Run different configurations to understand impact:

```bash
# Different synthetic ratios
# 1:1 real:synthetic
# 1:3 real:synthetic (default)
# 1:5 real:synthetic

# Filtered vs unfiltered synthetic
# With/without label smoothing
# With/without focal loss
```

---

## Complete Training Pipeline

### Step-by-Step

1. **Phase 1: Setup** (30 minutes)
   ```bash
   conda env create -f environment.yml
   python download_mvtec.py
   ```

2. **Phase 2: GAN Training** (2-3 days)
   ```bash
   python src/train_gan.py --config config.yaml
   ```

3. **Phase 3: Quality Control** (2 hours)
   ```bash
   python src/evaluate_quality.py --config config.yaml
   ```

4. **Phase 4: Classifier Training** (4 hours)
   ```bash
   python src/train_classifier.py --config config.yaml
   ```

5. **Phase 5: Visualization** (30 minutes)
   ```bash
   jupyter notebook notebooks/05_visualization.ipynb
   ```

### Total Time: ~3 days on RTX 4090

---

## Monitoring & Debugging

### Check Training Progress

```bash
# View logs
tail -f logs/gan-trainer.log

# Check GPU usage
nvidia-smi

# View wandb dashboard
# https://wandb.ai/your-username/gan-defect-augmentation
```

### Common Issues

#### Issue: Training diverges
**Solution**: Reduce learning rate, increase gradient penalty weight

#### Issue: Mode collapse (same images generated)
**Solution**: Increase discriminator steps, reduce learning rate

#### Issue: Poor quality images
**Solution**: Train longer, increase batch size, check data loading

#### Issue: Slow training
**Solution**: Enable mixed precision, increase num_workers, increase batch_size

---

## Performance Benchmarks

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 8GB | 24GB+ |
| CPU Cores | 4 | 8+ |
| RAM | 16GB | 32GB+ |
| Disk | 50GB | 100GB+ |

### Training Times (RTX 4090)

| Phase | Time | GPU Memory |
|-------|------|-----------|
| Phase 2: GAN | 2-3 days | 24GB |
| Phase 3: Filter | 2 hours | 8GB |
| Phase 4: Classifier | 4 hours | 12GB |
| **Total** | **~3 days** | - |

### Model Sizes

| Model | Parameters | Size |
|-------|-----------|------|
| Generator | ~50M | 200MB |
| Discriminator | ~30M | 120MB |
| Classifier | ~10M | 40MB |

---

## Next Steps

After completing Phases 2-4:

1. **Phase 5: Visualization**
   - Generate training curves
   - Create comparison plots
   - Build interactive dashboard
   - Generate PDF report

2. **Phase 6: Deployment**
   - Create FastAPI endpoints
   - Build Docker image
   - Deploy to Kubernetes
   - Setup monitoring

---

## References

### Papers
- WGAN-GP: https://arxiv.org/abs/1704.00028
- DT-GAN: https://arxiv.org/abs/2203.08270
- EfficientNet: https://arxiv.org/abs/1905.11946
- Focal Loss: https://arxiv.org/abs/1708.02002

### Tools
- PyTorch: https://pytorch.org
- timm: https://github.com/rwightman/pytorch-image-models
- wandb: https://wandb.ai
- Albumentations: https://albumentations.ai

---

## Support

- **Issues**: Check GitHub issues
- **Questions**: See README.md FAQ
- **Bugs**: Create issue with error log and config

---

**Status**: Phases 2-4 Implementation Complete ✅

Ready to run training pipeline!
