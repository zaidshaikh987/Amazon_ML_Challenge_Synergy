# 🚀 Complete Installation Guide

## ✅ Fixed: PyTorch Version Issue

**Problem**: PyTorch 2.1.2 is no longer available in the repository.

**Solution**: Updated to PyTorch 2.5.1 (latest stable version).

---

## 📦 Installation Steps

### Option 1: Automated Installation (Recommended)

Simply run:
```batch
cmd /c install_auto.bat
```

This will automatically:
1. ✅ Create virtual environment (if needed)
2. ✅ Upgrade pip, setuptools, wheel
3. ✅ Install PyTorch 2.5.1 (CPU version)
4. ✅ Install all remaining packages
5. ✅ Verify installation

**Time**: 5-10 minutes depending on internet speed

---

### Option 2: Manual Installation

```batch
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
venv\Scripts\activate.bat

# 3. Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# 4. Install PyTorch
pip install torch==2.5.1+cpu torchvision==0.20.1+cpu torchaudio==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu

# 5. Install core packages
pip install numpy==1.24.3 pandas==2.0.3 scipy==1.11.4

# 6. Install ML packages
pip install scikit-learn==1.3.2 lightgbm==4.1.0

# 7. Install transformers
pip install transformers==4.36.2 tokenizers==0.15.0 huggingface-hub==0.20.1 safetensors==0.4.1

# 8. Install sentence-transformers
pip install sentence-transformers==2.2.2

# 9. Install image processing
pip install Pillow==10.1.0 opencv-python==4.8.1.78

# 10. Install utilities
pip install requests==2.31.0 urllib3==2.1.0 certifi==2023.11.17 tqdm==4.66.1

# 11. Install visualization
pip install matplotlib==3.8.2 seaborn==0.13.0

# 12. Install additional dependencies
pip install regex==2023.12.25 filelock==3.13.1 packaging==23.2 pyyaml==6.0.1
pip install typing-extensions==4.9.0 sympy==1.12 joblib==1.3.2 threadpoolctl==3.2.0

# 13. Verify
python verify_setup.py
```

---

## 📋 Updated Package Versions

| Package | Old Version | New Version | Reason |
|---------|-------------|-------------|---------|
| torch | 2.1.2 | 2.5.1+cpu | Not available in repo |
| torchvision | 0.16.2 | 0.20.1+cpu | Matches PyTorch |
| torchaudio | 2.1.2 | 2.5.1+cpu | Matches PyTorch |

All other packages remain the same and are fully compatible.

---

## 🎮 GPU Support

If you have an NVIDIA GPU:

```batch
# Install PyTorch with CUDA 11.8
pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

Or for CUDA 12.1:
```batch
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

---

## ✅ Verify Installation

After installation completes, run:

```batch
python verify_setup.py
```

You should see:
```
✅ PyTorch: 2.5.1+cpu
✅ Transformers: 4.36.2
✅ Sentence-Transformers: 2.2.2
✅ LightGBM: 4.1.0
✅ NumPy: 1.24.3
✅ Pandas: 2.0.3
```

---

## 🔍 Troubleshooting

### PowerShell Execution Policy Error

If you see this error:
```
install_all.bat : The term 'install_all.bat' is not recognized
```

**Solution**: Use `cmd /c` prefix:
```powershell
cmd /c install_auto.bat
```

Or use `.\` prefix:
```powershell
.\install_auto.bat
```

### PyTorch Version Not Found

If you see:
```
ERROR: Could not find a version that satisfies the requirement torch==X.X.X
```

**Solution**: The scripts are now updated to use PyTorch 2.5.1 which is available. Run:
```batch
cmd /c install_auto.bat
```

### Virtual Environment Already Exists

If you want a clean installation:

```batch
# Remove old environment
rmdir /s /q venv

# Run installation again
cmd /c install_auto.bat
```

### Installation Hangs

If installation appears to hang:
- Be patient, PyTorch is ~200MB and takes time
- Check your internet connection
- Press Ctrl+C to cancel and try again

### Import Errors After Installation

```batch
# Activate environment first
venv\Scripts\activate.bat

# Then test imports
python -c "import torch; print(torch.__version__)"
```

---

## 📊 What Gets Installed

### Package Sizes (Approximate)

| Category | Size | Packages |
|----------|------|----------|
| PyTorch | ~200 MB | torch, torchvision, torchaudio |
| Transformers | ~50 MB | transformers, tokenizers, etc. |
| Data Science | ~100 MB | numpy, pandas, scipy, sklearn |
| Image Processing | ~30 MB | Pillow, opencv-python |
| ML | ~20 MB | lightgbm |
| Utilities | ~20 MB | requests, tqdm, matplotlib, etc. |
| **Total** | **~420 MB** | All packages |

### Disk Space Requirements

- Virtual environment: ~550 MB
- Models cache (first run): ~2-5 GB
- Images: Variable (depends on dataset)
- **Total recommended**: 10 GB free space

---

## 🚀 After Installation

### 1. Test the Setup

```batch
python verify_setup.py
```

### 2. Run Sample Pipeline

```batch
cmd /c run_sample.bat
```

This will:
- Download sample images
- Extract embeddings
- Train a model
- Generate predictions

### 3. Start Development

```batch
# Activate environment
venv\Scripts\activate.bat

# Run your code
python src/extract_embeddings.py --help
python src/train_baseline.py --help
python src/predict.py --help
```

---

## 📚 Additional Resources

- **Quick Start**: See `QUICKSTART.md`
- **Full Documentation**: See `README.md`
- **Package Details**: See `PACKAGE_COMPATIBILITY.md`
- **Environment Setup**: See `ENVIRONMENT_SETUP.md`

---

## ⚙️ System Requirements

### Minimum
- Python 3.8+
- 8 GB RAM
- 10 GB free disk space
- Windows 10/11, Ubuntu 20.04+, or macOS 12+

### Recommended
- Python 3.9 or 3.10
- 16 GB RAM
- 20 GB free disk space
- NVIDIA GPU with 6GB+ VRAM (optional, for faster training)

---

## 🎯 Next Steps

1. ✅ Virtual environment created
2. ✅ All packages installed
3. ✅ Installation verified
4. 🎯 **Run sample pipeline**: `cmd /c run_sample.bat`
5. 🎯 **Start training**: See README.md for full pipeline

---

## 📞 Need Help?

If you encounter issues:

1. Check `ENVIRONMENT_SETUP.md` troubleshooting section
2. Verify Python version: `python --version` (should be 3.8-3.11)
3. Check pip version: `pip --version`
4. Try clean reinstall (remove venv and run again)
5. Check PyTorch official docs: https://pytorch.org/get-started/locally/

---

**Last Updated**: October 11, 2025  
**PyTorch Version**: 2.5.1  
**Status**: ✅ Ready for Installation
