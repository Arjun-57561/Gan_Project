# Textile Defect Detection — Complete Project Explanation
> Based on your actual codebase in `src/`

---

## STEP 1 — DATA UNDERSTANDING

Your project uses the **MVTec AD dataset** — a real industrial anomaly detection benchmark.

**Dataset structure (from `mvtec_dataset.py`):**
```
data/raw/mvtec/
  bottle/
    train/
      good/         ← normal images only (label = 0)
    test/
      good/         ← normal test images
      broken_large/ ← defect type 1 (label = 1)
      broken_small/ ← defect type 2 (label = 1)
      contamination/← defect type 3 (label = 1)
    ground_truth/   ← pixel-level defect masks (_mask.png)
```

Your code handles **15 categories**: bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper.

**Class imbalance reality:**
- Training set contains ONLY normal (good) images — this is by design in MVTec
- Defective images appear only in the test split
- This is exactly why your project uses GAN — to generate synthetic defect images for training

**Why preprocessing is required:**
- Images come in varying sizes across categories
- Pixel values need to be in a consistent range for neural networks
- Raw images contain noise and lighting variations that can mislead the model

---

## STEP 2 — DATA PREPROCESSING

Your preprocessing is implemented in `src/data/transforms.py` using the **Albumentations** library.

**Training transforms (`get_train_transforms`):**
```python
A.Resize(256, 256)                          # Fixed size for all images
A.HorizontalFlip(p=0.5)                     # Random horizontal flip
A.Rotate(limit=15, p=0.5)                   # Small rotation ±15°
A.RandomBrightnessContrast(0.2, 0.2, p=0.5) # Lighting variation
A.GaussNoise(p=0.2)                         # Noise robustness
A.Normalize(mean=[0.485, 0.456, 0.406],     # ImageNet normalization
            std=[0.229, 0.224, 0.225])
ToTensorV2()                                # Convert to PyTorch tensor
```

**Why each step matters:**
- `Resize(256, 256)` — your Generator and Discriminator are built for 256×256 input
- `Normalize` with ImageNet stats — your classifier uses EfficientNet-B2 pretrained on ImageNet, so same normalization is required
- Augmentations (flip, rotate, brightness) — artificially expand the limited training data
- `GaussNoise` — makes the model robust to sensor noise in real factory cameras

**Corrupted image handling:**
- `Image.open(img_path).convert("RGB")` in `mvtec_dataset.py` — the `.convert("RGB")` call handles grayscale or RGBA images automatically
- Missing masks default to a zero array: `mask = np.zeros(...)` so the pipeline never crashes

**Validation/Test transforms (`get_val_transforms`, `get_test_transforms`):**
- Only Resize + Normalize — no augmentation, for fair evaluation

---

## STEP 3 — GAN DATASET CREATION

In MVTec, the training split contains **only normal/good images**. Your GAN is trained on these normal images and learns to synthesize defects on top of them.

**How it works in your code (`train_gan.py`):**
```python
# Each batch contains:
real_images  = batch["image"]   # Normal textile image (3 channels)
defect_masks = batch["mask"]    # Where to place the defect (1 channel)
defect_types = batch["label"]   # What type of defect (0-14)
```

The Generator takes a normal image + a defect mask + a defect type index, and outputs a realistic defective version of that image.

**Why not train GAN on normal images alone:**
- The goal is not to generate random images — it's to generate *controlled defects* at specific locations
- Training on defect images + masks teaches the GAN the spatial structure of real defects
- The defect type embedding (`nn.Embedding(15, 64)`) lets the GAN learn 15 different defect styles

---

## STEP 4 — DCGAN CONCEPT

Your project uses a more advanced architecture than basic DCGAN — it's a **conditional U-Net GAN with spectral normalization**.

**Generator (`src/models/generator.py`):**
- Takes: normal image (3ch) + defect mask (1ch) + defect type index
- Architecture: U-Net encoder-decoder with skip connections
- Encoder: 4 downsampling blocks (256→128→64→32→16)
- Bottleneck: ResidualBlock with Conditional Instance Normalization
- Decoder: 4 upsampling blocks with skip connections from encoder
- Output: synthetic defective image (3ch, same size as input)
- Final activation: `Tanh()` → output in range [-1, 1]

**Discriminator (`src/models/discriminator.py`):**
- Takes: image (3ch) + defect mask (1ch) = 4 channels
- Architecture: Multi-scale PatchGAN (3 scales)
- Each scale uses 6 strided convolutions with LeakyReLU
- Output: patch-level validity scores (not a single scalar)
- Multi-scale: evaluates at 256×256, 128×128, and 64×64

**Adversarial training:**
- Generator tries to fool the Discriminator into thinking fake images are real
- Discriminator tries to correctly distinguish real from fake
- They compete — Generator improves until Discriminator can't tell the difference

**Why this is better than basic DCGAN:**
- Spectral normalization stabilizes training
- U-Net skip connections preserve fine texture details
- Multi-scale discriminator catches both global structure and local texture
- Conditional on defect type — generates specific defect patterns, not random noise

---

## STEP 5 — GAN TRAINING PROCESS

Your training loop is in `GANTrainer` class (`src/train_gan.py`).

**Step-by-step per batch:**

1. **Discriminator training** (runs 5 times per generator update — `discriminator_steps: 5`):
   ```
   real_images → Discriminator → d_real score
   Generator(normal, mask, type) → fake_images
   fake_images → Discriminator → d_fake score
   loss_d = -d_real.mean() + d_fake.mean() + 10.0 × gradient_penalty
   ```

2. **Generator training** (runs 1 time):
   ```
   Generator(normal, mask, type) → fake_images
   fake_images → Discriminator → d_fake score
   loss_g = -d_fake.mean()   ← Generator wants Discriminator to say "real"
   ```

3. **Loss function — WGAN-GP (Wasserstein GAN with Gradient Penalty):**
   - Not BCELoss — your project uses Wasserstein distance
   - Gradient penalty weight = 10.0 (from config)
   - More stable than vanilla GAN, avoids mode collapse

4. **Optimization:**
   - Adam optimizer with β1=0.0, β2=0.9 (standard for WGAN)
   - Generator LR: 1e-4, Discriminator LR: 2e-4
   - Mixed precision (FP16) via `GradScaler` for speed
   - Gradient clipping at 1.0 to prevent exploding gradients

5. **Training duration:** 200 epochs, early stopping patience of 20 epochs

---

## STEP 6 — GAN OUTPUT & ANALYSIS

Your quality evaluation is in `src/evaluate_quality.py` — `QualityEvaluator` class.

**Metrics used to evaluate generated images:**

| Metric | What it measures | Weight |
|--------|-----------------|--------|
| FID (Fréchet Inception Distance) | Statistical similarity between real and fake distributions | 30% |
| LPIPS | Perceptual similarity (texture, structure) | 30% |
| Defect Coverage (IoU) | How well synthetic defect matches the mask | 20% |
| Sharpness (Laplacian variance) | Image clarity, not blurry | 20% |

**Why early outputs look like noise:**
- The Generator starts with random weights — it produces random pixel patterns
- The Discriminator also starts random — it can't distinguish real from fake yet
- Over epochs, the Generator learns to produce textures that fool the Discriminator
- Skip connections in the U-Net help preserve the original image structure early on

**Quality filtering:**
```python
# Only top 50% of generated images are kept (keep_ratio: 0.5 in config)
df_scores.sort_values("final", ascending=False).head(num_keep)
```

---

## STEP 7 — DATA AUGMENTATION USING GAN

After GAN training, synthetic defective images are combined with real images to train the classifier.

**Your augmentation pipeline:**
- Real normal images: from MVTec train/good/
- Synthetic defective images: generated by your trained Generator
- Combined dataset fed to EfficientNet-B2 classifier

**Impact on training:**
- Without GAN: classifier trained only on normal images → poor defect detection
- With GAN: classifier sees diverse defect patterns → better generalization
- The 15-class defect type embedding ensures variety across defect types

**Training transforms in `transforms.py` also augment:**
- Horizontal flip, rotation, brightness/contrast, Gaussian noise
- These are applied on top of GAN-generated images for even more diversity

---

## STEP 8 — DEFECT DETECTION MODEL

Your classifier is in `src/train_classifier.py` using **EfficientNet-B2** via the `timm` library.

**Transfer learning setup:**
```python
self.model = timm.create_model(
    "efficientnet_b2",   # Pretrained on ImageNet
    pretrained=True,     # Load ImageNet weights
    num_classes=15,      # Replace final layer for 15 MVTec categories
)
```

**Why EfficientNet-B2:**
- Pretrained on 1.2M ImageNet images — already knows edges, textures, shapes
- EfficientNet scales depth/width/resolution together — efficient and accurate
- B2 variant: good balance of accuracy vs. compute for 256×256 images

**Loss function — Focal Loss:**
```python
focal_loss = alpha × (1 - p)^gamma × cross_entropy
# alpha=0.25, gamma=2.0
```
- Focal Loss down-weights easy examples (normal images) and focuses on hard ones (rare defect types)
- Directly addresses class imbalance

**Training details:**
- AdamW optimizer, LR=1e-3
- CosineAnnealingLR scheduler over 100 epochs
- Test-Time Augmentation (TTA) with 4 augmentations for better predictions
- Mixed precision training for speed

---

## STEP 9 — FINAL PIPELINE

```
1. RAW DATA
   MVTec AD dataset → 15 categories, normal + defective images + pixel masks

2. PREPROCESSING (transforms.py + mvtec_dataset.py)
   → Resize to 256×256
   → Normalize with ImageNet stats
   → Augment (flip, rotate, brightness, noise)
   → Load as PyTorch tensors with DataLoader

3. GAN TRAINING (train_gan.py)
   Input: normal image + defect mask + defect type
   Generator (U-Net) ←→ Discriminator (Multi-scale PatchGAN)
   Loss: WGAN-GP (Wasserstein + Gradient Penalty)
   200 epochs, 5 discriminator steps per generator step

4. SYNTHETIC IMAGE GENERATION
   Trained Generator produces defective images
   Quality evaluated: FID + LPIPS + Coverage + Sharpness
   Top 50% kept (quality_control.keep_ratio = 0.5)

5. DATASET AUGMENTATION
   Real normal images + Synthetic defective images = augmented training set

6. CLASSIFIER TRAINING (train_classifier.py)
   EfficientNet-B2 (pretrained) fine-tuned on augmented dataset
   Focal Loss for class imbalance
   100 epochs with CosineAnnealingLR

7. FINAL OUTPUT
   Binary classification: normal (0) vs defective (1)
   Metrics: Accuracy, F1-macro, F1-weighted, AUC-ROC
```

---

## STEP 10 — PROJECT JUSTIFICATION

**The problem:**
- In real textile factories, defects are rare — maybe 1-5% of production
- MVTec training set has ZERO defective images (only normal images in train split)
- A classifier trained only on normal images cannot learn what defects look like

**Why traditional methods fail:**
- Simple thresholding: fails on complex textures (carpet, leather, grid)
- Classical CV (edge detection, morphology): can't generalize across 15 defect types
- Training CNN from scratch: not enough defect data → overfitting

**How GAN solves it:**
- Generator learns the distribution of defect patterns from test images + masks
- Produces unlimited synthetic defective images at training time
- Defect type conditioning (15 classes) ensures diverse, realistic outputs
- Quality filtering (FID, LPIPS, IoU) ensures only realistic images are used

**Result:**
- Classifier trained on GAN-augmented data sees realistic defect patterns
- Better generalization to unseen defects in production

---

## STEP 11 — NOVELTY

**What makes your project novel:**

1. **Conditional defect synthesis** — not just random image generation. Your Generator takes a defect mask as spatial guidance, so defects appear exactly where they should (scratch on surface, not in background).

2. **15-class defect type embedding** — a single GAN model handles all 15 MVTec categories using a learned embedding (`nn.Embedding(15, 64)`), rather than training 15 separate GANs.

3. **Multi-scale discrimination** — PatchGAN at 3 scales (256, 128, 64) catches both macro-level realism and micro-level texture quality.

4. **Automated quality control pipeline** — generated images are not blindly used. `evaluate_quality.py` scores each image on 4 metrics and filters out low-quality ones before augmentation.

5. **WGAN-GP training** — more stable than vanilla GAN, avoids mode collapse which is a common failure in defect generation.

---

## STEP 12 — REAL-WORLD APPLICATION

**Direct industry applications:**

- **Textile manufacturing**: Automated inline inspection on production lines — cameras capture fabric, model flags defects in real-time before products reach packaging.

- **Quality control automation**: Replaces manual visual inspection (slow, expensive, inconsistent) with 24/7 automated detection.

- **Defect database building**: GAN generates diverse synthetic defects for training future models — no need to wait for rare real defects to accumulate.

- **Multi-product adaptability**: Your 15-category model covers bottle, cable, carpet, leather, metal, wood, etc. — one system for multiple product lines.

- **Cost reduction**: Early defect detection prevents defective products from reaching customers → reduces returns, rework, and waste.

- **Scalability**: Once trained, the model runs on standard GPU hardware at production line speeds.

**Your config shows production-ready thinking:**
- `num_workers: 8` — parallel data loading for speed
- `mixed_precision: true` — FP16 for faster inference
- `wandb_project` — experiment tracking for model versioning
- Checkpoint saving — model can be deployed and updated incrementally
