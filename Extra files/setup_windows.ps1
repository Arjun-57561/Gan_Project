# GAN Defect Augmentation - Windows PowerShell Setup Script
# Run this script to fix all errors in 10 minutes

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GAN Defect Augmentation - Windows Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as admin (optional but recommended)
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
if (-not $isAdmin) {
    Write-Host "⚠️  Not running as admin (optional, but recommended)" -ForegroundColor Yellow
}

# Step 1: Check Python
Write-Host "Step 1: Checking Python..." -ForegroundColor Green
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "❌ Python not found. Install from https://www.python.org/" -ForegroundColor Red
    exit 1
}

# Step 2: Create virtual environment
Write-Host ""
Write-Host "Step 2: Creating virtual environment..." -ForegroundColor Green
if (Test-Path "gan_env") {
    Write-Host "✅ Virtual environment already exists" -ForegroundColor Green
} else {
    python -m venv gan_env
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Virtual environment created" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}

# Step 3: Activate virtual environment
Write-Host ""
Write-Host "Step 3: Activating virtual environment..." -ForegroundColor Green
& ".\gan_env\Scripts\Activate.ps1"
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "⚠️  Activation may have issues, continuing..." -ForegroundColor Yellow
}

# Step 4: Upgrade pip
Write-Host ""
Write-Host "Step 4: Upgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip -q
Write-Host "✅ Pip upgraded" -ForegroundColor Green

# Step 5: Install PyTorch with CUDA
Write-Host ""
Write-Host "Step 5: Installing PyTorch with CUDA 12.1..." -ForegroundColor Green
Write-Host "⏳ This may take 2-3 minutes..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ PyTorch installed" -ForegroundColor Green
} else {
    Write-Host "❌ PyTorch installation failed" -ForegroundColor Red
    exit 1
}

# Step 6: Install ML packages
Write-Host ""
Write-Host "Step 6: Installing ML packages..." -ForegroundColor Green
Write-Host "⏳ This may take 1-2 minutes..." -ForegroundColor Yellow
pip install wandb omegaconf albumentations timm pytorch-fid scikit-learn pandas matplotlib seaborn plotly tqdm rich tensorboard requests -q
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ ML packages installed" -ForegroundColor Green
} else {
    Write-Host "❌ ML packages installation failed" -ForegroundColor Red
    exit 1
}

# Step 7: Create directories
Write-Host ""
Write-Host "Step 7: Creating directories..." -ForegroundColor Green
New-Item -ItemType Directory -Path "data\raw" -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path "data\processed" -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path "checkpoints" -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path "logs" -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path "outputs" -ErrorAction SilentlyContinue | Out-Null
Write-Host "✅ Directories created" -ForegroundColor Green

# Step 8: Test imports
Write-Host ""
Write-Host "Step 8: Testing imports..." -ForegroundColor Green
python -c "import torch, wandb, omegaconf, albumentations, timm; print('✅ All imports successful')" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ All packages imported successfully" -ForegroundColor Green
} else {
    Write-Host "❌ Import test failed" -ForegroundColor Red
    exit 1
}

# Step 9: Test GPU
Write-Host ""
Write-Host "Step 9: Testing GPU..." -ForegroundColor Green
$gpuTest = python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')" 2>&1
Write-Host $gpuTest -ForegroundColor Green

# Step 10: Download dataset
Write-Host ""
Write-Host "Step 10: Downloading MVTec AD dataset..." -ForegroundColor Green
Write-Host "⏳ This may take 5-10 minutes (2.2 GB)..." -ForegroundColor Yellow
python download_mvtec_fixed.py
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Dataset downloaded" -ForegroundColor Green
} else {
    Write-Host "⚠️  Dataset download had issues (can retry later)" -ForegroundColor Yellow
}

# Final summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✅ SETUP COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Green
Write-Host "1. Activate environment: .\gan_env\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "2. Test setup: python src/main.py --config config.yaml" -ForegroundColor White
Write-Host "3. Train GAN: python src/train_gan.py --config config.yaml" -ForegroundColor White
Write-Host ""
Write-Host "Documentation:" -ForegroundColor Green
Write-Host "- START_HERE.md - Quick start" -ForegroundColor White
Write-Host "- README.md - Full documentation" -ForegroundColor White
Write-Host "- QUICKSTART.md - 5-minute guide" -ForegroundColor White
Write-Host ""
