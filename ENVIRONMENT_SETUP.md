# Virtual Environment Setup Guide

This guide will help you set up a conflict-free Python virtual environment with all required packages for the Amazon ML Challenge project.

## 🎯 Quick Setup (Recommended)

### Windows
```batch
# Run the automated setup script
setup_venv.bat
```

### Linux/Mac
```bash
# Make script executable
chmod +x setup_venv.sh

# Run the setup script
./setup_venv.sh
```

The script will:
1. Check Python installation
2. Ask if you want GPU (CUDA) support
3. Create a fresh virtual environment
4. Install all packages with compatible versions
5. Verify the installation

---

## 📦 What Gets Installed

### Core ML Stack
- **PyTorch 2.1.2** - Deep learning framework
  - CPU version by default
  - Optional CUDA 11.8 version for GPU
- **Transformers 4.36.2** - Hugging Face models
- **Sentence-Transformers 2.2.2** - Text embeddings
- **LightGBM 4.1.0** - Gradient boosting

### Data Science
- **NumPy 1.24.3** - Numerical computing
- **Pandas 2.0.3** - Data manipulation
- **Scikit-learn 1.3.2** - ML utilities
- **SciPy 1.11.4** - Scientific computing

### Image Processing
- **Pillow 10.1.0** - Image handling
- **OpenCV 4.8.1.78** - Computer vision

### Utilities
- **Requests 2.31.0** - HTTP requests
- **tqdm 4.66.1** - Progress bars
- **Matplotlib 3.8.2** - Visualization
- **Seaborn 0.13.0** - Statistical plots

---

## 🔧 Manual Setup (Alternative)

If you prefer manual setup:

### Step 1: Create Virtual Environment
```bash
# Create venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate.bat

# Activate (Linux/Mac)
source venv/bin/activate
```

### Step 2: Upgrade pip
```bash
python -m pip install --upgrade pip setuptools wheel
```

### Step 3: Install PyTorch

**For CPU only:**
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
```

**For GPU (CUDA 11.8):**
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

**For GPU (CUDA 12.1):**
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

### Step 4: Install Remaining Packages
```bash
# Core packages
pip install numpy==1.24.3 pandas==2.0.3 scipy==1.11.4
pip install scikit-learn==1.3.2 lightgbm==4.1.0

# NLP packages
pip install transformers==4.36.2 sentence-transformers==2.2.2
pip install tokenizers==0.15.0 huggingface-hub==0.20.1 safetensors==0.4.1

# Image processing
pip install Pillow==10.1.0 opencv-python==4.8.1.78

# Utilities
pip install requests==2.31.0 tqdm==4.66.1
pip install matplotlib==3.8.2 seaborn==0.13.0

# Additional dependencies
pip install regex==2023.12.25 filelock==3.13.1 packaging==23.2
pip install pyyaml==6.0.1 typing-extensions==4.9.0 sympy==1.12
pip install joblib==1.3.2 threadpoolctl==3.2.0
```

---

## ✅ Verify Installation

After setup, verify everything is working:

```python
# Test imports
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import sentence_transformers; print(f'Sentence-Transformers: {sentence_transformers.__version__}')"
python -c "import lightgbm; print(f'LightGBM: {lightgbm.__version__}')"

# Check GPU (if installed)
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

Or run the verification script:
```bash
python verify_setup.py
```

---

## 🎮 GPU Support

### Check CUDA Version
```bash
# Windows
nvidia-smi

# Linux
nvidia-smi
nvcc --version
```

### CUDA Compatibility
- **CUDA 11.8** → Use `requirements-gpu.txt` or setup script
- **CUDA 12.1+** → Modify installation command (see Manual Setup)
- **No GPU** → Use `requirements.txt` (CPU only)

---

## 🐛 Troubleshooting

### Issue: "pip is not recognized"
**Solution:** 
```bash
# Windows
python -m pip install --upgrade pip

# Linux/Mac
python3 -m pip install --upgrade pip
```

### Issue: PyTorch CUDA version mismatch
**Solution:** Reinstall PyTorch with correct CUDA version
```bash
# Check your CUDA version first
nvidia-smi

# Then install matching PyTorch version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "No module named 'sentence_transformers'"
**Solution:** Install in correct order
```bash
pip install torch  # Install PyTorch first
pip install transformers
pip install sentence-transformers
```

### Issue: OpenCV import error
**Solution:** Reinstall opencv-python
```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python==4.8.1.78
```

### Issue: LightGBM installation fails
**Solution:**
- **Windows:** Install Microsoft C++ Build Tools
- **Linux:** `sudo apt-get install build-essential`
- **Mac:** `xcode-select --install`

### Issue: Permission denied (Linux/Mac)
**Solution:**
```bash
# Don't use sudo with pip in venv
# Make sure venv is activated first
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🔄 Update Packages

To update packages to newer compatible versions:

```bash
# Activate venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate.bat  # Windows

# Update pip
python -m pip install --upgrade pip

# Update specific package
pip install --upgrade package-name

# Or update all from requirements
pip install --upgrade -r requirements.txt
```

---

## 🗑️ Clean Reinstall

If you encounter issues, do a clean reinstall:

```bash
# Deactivate if active
deactivate

# Remove old environment
# Windows
rmdir /s /q venv

# Linux/Mac
rm -rf venv

# Run setup script again
setup_venv.bat  # Windows
./setup_venv.sh  # Linux/Mac
```

---

## 📊 System Requirements

### Minimum
- Python 3.8+
- 8GB RAM
- 10GB disk space

### Recommended
- Python 3.9 or 3.10
- 16GB RAM
- 20GB disk space
- NVIDIA GPU with 6GB+ VRAM (optional)

### Tested On
- ✅ Windows 10/11
- ✅ Ubuntu 20.04/22.04
- ✅ macOS 12+ (Intel & Apple Silicon)
- ✅ Python 3.8, 3.9, 3.10, 3.11

---

## 🚀 Next Steps

After setup is complete:

1. **Test the environment**
   ```bash
   python verify_setup.py
   ```

2. **Run sample pipeline**
   ```bash
   run_sample.bat  # Windows
   ./run_sample.sh  # Linux/Mac
   ```

3. **Start development**
   ```bash
   # Activate environment
   venv\Scripts\activate.bat  # Windows
   source venv/bin/activate    # Linux/Mac
   
   # Run your code
   python src/extract_embeddings.py --help
   ```

---

## 📝 Notes

- **Package Versions**: All versions are tested for compatibility
- **No Conflicts**: Carefully selected to avoid dependency conflicts
- **CPU by Default**: GPU support is optional
- **Offline Install**: Download wheels and use `pip install --no-index`
- **Requirements Files**:
  - `requirements.txt` - CPU only
  - `requirements-gpu.txt` - With CUDA 11.8

---

## 📞 Support

If you encounter issues:

1. Check this troubleshooting guide
2. Verify Python version: `python --version`
3. Check pip version: `pip --version`
4. Review error messages carefully
5. Try clean reinstall
6. Check PyTorch compatibility: https://pytorch.org/get-started/locally/

---

**Last Updated:** October 11, 2025  
**Compatible With:** Python 3.8 - 3.11  
**PyTorch Version:** 2.1.2
