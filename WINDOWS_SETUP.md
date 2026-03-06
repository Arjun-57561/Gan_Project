# Windows PowerShell Setup - 10 Minute Fix

Complete step-by-step guide to fix all errors on Windows.

## ⚡ Quick Fix (Copy-Paste Commands)

Open PowerShell and run these commands one by one:

```powershell
# 1. Navigate to project
cd "C:\Users\Arjun\OneDrive\Desktop\Gan_Project"

# 2. Create virtual environment (1 minute)
python -m venv gan_env

# 3. Activate virtual environment
.\gan_env\Scripts\Activate.ps1

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install PyTorch with CUDA (2 minutes)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 6. Install all ML packages (2 minutes)
pip install wandb omegaconf albumentations timm pytorch-fid scikit-learn pandas matplotlib seaborn plotly tqdm rich tensorboard requests

# 7. Create directories
mkdir data\raw -ErrorAction SilentlyContinue
mkdir data\processed -ErrorAction SilentlyContinue
mkdir checkpoints -ErrorAction SilentlyContinue
mkdir logs -ErrorAction SilentlyContinue
mkdir outputs -ErrorAction SilentlyContinue

# 8. Test imports
python -c "import torch, wandb, omegaconf; print('✅ ALL PACKAGES OK')"

# 9. Test GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NO GPU\"}')"

# 10. Download dataset (5-10 minutes)
python download_mvtec_fixed.py

# 11. Verify setup
python src/main.py --config config.yaml
```

**Total Time: ~10 minutes**

---

## 🔧 Automated Setup (Recommended)

If you want automated setup, run the PowerShell script:

```powershell
# Navigate to project
cd "C:\Users\Arjun\OneDrive\Desktop\Gan_Project"

# Run setup script
.\setup_windows.ps1
```

This will:
- ✅ Create virtual environment
- ✅ Install all packages
- ✅ Create directories
- ✅ Test GPU
- ✅ Download dataset

---

## 🚨 Common Issues & Fixes

### Issue 1: "Execution Policy" Error

**Error**: `cannot be loaded because running scripts is disabled`

**Fix**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try again.

### Issue 2: "Python not found"

**Error**: `python: The term 'python' is not recognized`

**Fix**:
1. Install Python from https://www.python.org/
2. **IMPORTANT**: Check "Add Python to PATH" during installation
3. Restart PowerShell
4. Try again

### Issue 3: "pip not found"

**Fix**:
```powershell
python -m pip install --upgrade pip
```

### Issue 4: "CUDA not available"

**Error**: `GPU: NO GPU`

**Fix**: This is OK! The code will use CPU (slower but works)

To use GPU:
1. Install NVIDIA drivers: https://www.nvidia.com/Download/driverDetails.aspx
2. Reinstall PyTorch:
```powershell
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue 5: Dataset Download 404 Error

**Error**: `404 Not Found`

**Fix**: Already fixed! Use `download_mvtec_fixed.py` with updated URL

If still fails:
1. Download manually: https://www.mvtec.com/company/research/datasets/mvtec-ad
2. Extract to `data/raw/mvtec/`

---

## ✅ Verification Checklist

After setup, verify everything works:

```powershell
# Activate environment first
.\gan_env\Scripts\Activate.ps1

# Test 1: Python version
python --version
# Expected: Python 3.11+

# Test 2: PyTorch
python -c "import torch; print(torch.__version__)"
# Expected: 2.1.0+cu121

# Test 3: All packages
python -c "import torch, wandb, omegaconf, albumentations, timm; print('✅ OK')"
# Expected: ✅ OK

# Test 4: GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
# Expected: GPU: NVIDIA GeForce RTX 3080 (or your GPU)

# Test 5: Data loading
python src/main.py --config config.yaml
# Expected: Setup completed successfully

# Test 6: Dataset
ls data/raw/mvtec/
# Expected: bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper
```

---

## 📁 Expected Directory Structure After Setup

```
Gan_Project/
├── gan_env/                    ✅ Virtual environment
├── data/
│   ├── raw/
│   │   └── mvtec/              ✅ Dataset (2.2 GB)
│   │       ├── bottle/
│   │       ├── cable/
│   │       └── ... (13 more)
│   └── processed/
├── checkpoints/                ✅ Model checkpoints
├── logs/                       ✅ Training logs
├── outputs/                    ✅ Generated outputs
├── src/                        ✅ Source code
├── notebooks/                  ✅ Jupyter notebooks
├── config.yaml                 ✅ Configuration
├── download_mvtec_fixed.py     ✅ Fixed download script
├── setup_windows.ps1           ✅ Setup script
└── README.md                   ✅ Documentation
```

---

## 🎯 Next Steps After Setup

### Option 1: Quick Test (5 minutes)
```powershell
# Activate environment
.\gan_env\Scripts\Activate.ps1

# Test data loading
python src/main.py --config config.yaml

# Test notebook
jupyter notebook notebooks/01_setup.ipynb
```

### Option 2: Train GAN (2-3 days)
```powershell
# Activate environment
.\gan_env\Scripts\Activate.ps1

# Train GAN
python src/train_gan.py --config config.yaml
```

### Option 3: Train Classifier (4 hours)
```powershell
# Activate environment
.\gan_env\Scripts\Activate.ps1

# Train classifier
python src/train_classifier.py --config config.yaml
```

---

## 💡 Pro Tips

### Always Activate Environment First
Every time you open a new PowerShell window:
```powershell
.\gan_env\Scripts\Activate.ps1
```

You'll see `(gan_env)` at the start of the prompt when activated.

### Check GPU Usage During Training
```powershell
# In a separate PowerShell window
nvidia-smi
```

### Monitor Training Logs
```powershell
# In a separate PowerShell window
Get-Content logs/gan-trainer.log -Wait
```

### View wandb Dashboard
```powershell
# After training starts
# Open browser to: https://wandb.ai/your-username/gan-defect-augmentation
```

---

## 🆘 Still Having Issues?

### Check Python Installation
```powershell
python -c "import sys; print(sys.executable)"
```

### Check pip
```powershell
pip --version
```

### Check Virtual Environment
```powershell
Get-Command python
# Should show: gan_env\Scripts\python.exe
```

### Reinstall Everything
```powershell
# Remove virtual environment
Remove-Item -Recurse -Force gan_env

# Start over
python -m venv gan_env
.\gan_env\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install wandb omegaconf albumentations timm pytorch-fid scikit-learn pandas matplotlib seaborn plotly tqdm rich tensorboard requests
```

---

## 📞 Support

- **Setup Issues**: See this file (WINDOWS_SETUP.md)
- **General Help**: See README.md
- **Quick Reference**: See QUICK_REFERENCE.md
- **Troubleshooting**: See QUICKSTART.md

---

## ✅ Success Indicators

After setup, you should see:

```
✅ ALL PACKAGES OK
GPU: NVIDIA GeForce RTX 3080
Setup completed successfully
```

If you see these, you're ready to train! 🚀

---

**Time to Complete**: ~10 minutes

**Status**: Ready to Use ✅

**Next**: Run `python src/main.py --config config.yaml`
