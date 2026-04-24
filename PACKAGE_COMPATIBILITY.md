# 📦 Package Compatibility Matrix

This document details the tested and verified package versions for conflict-free installation.

## ✅ Tested Configuration

| Package | Version | Purpose | License |
|---------|---------|---------|---------|
| **Python** | 3.8 - 3.11 | Runtime | PSF |
| **PyTorch** | 2.1.2 | Deep Learning | BSD-3 |
| **Transformers** | 4.36.2 | NLP Models | Apache-2.0 |
| **Sentence-Transformers** | 2.2.2 | Text Embeddings | Apache-2.0 |
| **LightGBM** | 4.1.0 | Gradient Boosting | MIT |
| **NumPy** | 1.24.3 | Numerical Computing | BSD-3 |
| **Pandas** | 2.0.3 | Data Manipulation | BSD-3 |
| **Scikit-learn** | 1.3.2 | ML Utilities | BSD-3 |
| **Pillow** | 10.1.0 | Image Processing | HPND |
| **OpenCV** | 4.8.1.78 | Computer Vision | Apache-2.0 |

## 🔗 Dependency Resolution

### Critical Dependencies

#### PyTorch Stack
```
torch==2.1.2
├── numpy>=1.21.0
├── typing-extensions>=4.8.0
├── sympy>=1.11.0
├── filelock>=3.13.0
└── jinja2>=3.0.0

torchvision==0.16.2
├── torch==2.1.2
├── numpy>=1.21.0
└── pillow>=8.0.0
```

#### Transformers Stack
```
transformers==4.36.2
├── torch>=1.11.0
├── numpy>=1.17.0
├── tokenizers>=0.14.0,<0.19.0
├── huggingface-hub>=0.19.0,<1.0.0
├── safetensors>=0.3.1
├── pyyaml>=5.1.0
├── regex>=2022.1.18
└── packaging>=20.0

sentence-transformers==2.2.2
├── transformers>=4.6.0,<5.0.0
├── torch>=1.6.0
├── numpy>=1.18.0
└── scikit-learn>=0.19.1
```

#### Data Science Stack
```
scikit-learn==1.3.2
├── numpy>=1.17.3,<2.0.0
├── scipy>=1.5.0
├── joblib>=1.1.1
└── threadpoolctl>=2.0.0

lightgbm==4.1.0
├── numpy>=1.17.0
└── scipy>=1.17.0

pandas==2.0.3
├── numpy>=1.20.3
├── python-dateutil>=2.8.2
└── pytz>=2020.1
```

## 🔍 Version Compatibility

### Python Version Support

| Python | PyTorch 2.1.2 | Transformers 4.36.2 | LightGBM 4.1.0 | Status |
|--------|---------------|---------------------|----------------|---------|
| 3.8 | ✅ | ✅ | ✅ | ✅ Tested |
| 3.9 | ✅ | ✅ | ✅ | ✅ Tested |
| 3.10 | ✅ | ✅ | ✅ | ✅ Tested |
| 3.11 | ✅ | ✅ | ✅ | ✅ Tested |
| 3.12 | ⚠️ | ⚠️ | ⚠️ | ⚠️ Limited |

### CUDA Compatibility

| CUDA Version | PyTorch 2.1.2 | Install Command |
|--------------|---------------|-----------------|
| 11.8 | ✅ Official | `--index-url https://download.pytorch.org/whl/cu118` |
| 12.1 | ✅ Official | `--index-url https://download.pytorch.org/whl/cu121` |
| CPU Only | ✅ Official | `--index-url https://download.pytorch.org/whl/cpu` |

### OS Compatibility

| OS | Status | Notes |
|----|--------|-------|
| **Windows 10/11** | ✅ Tested | Use `setup_venv.bat` |
| **Ubuntu 20.04** | ✅ Tested | Use `setup_venv.sh` |
| **Ubuntu 22.04** | ✅ Tested | Use `setup_venv.sh` |
| **macOS 12+ (Intel)** | ✅ Tested | CPU only |
| **macOS 12+ (M1/M2)** | ⚠️ Limited | MPS support experimental |

## ⚠️ Known Conflicts

### Conflict 1: NumPy Version
**Issue**: NumPy 2.0+ breaks compatibility with older packages

**Resolution**:
```bash
# Use NumPy 1.24.3
pip install "numpy<2.0.0"
```

### Conflict 2: Tokenizers Version
**Issue**: Transformers 4.36.2 requires tokenizers <0.19.0

**Resolution**:
```bash
# Install specific tokenizers version
pip install "tokenizers>=0.14.0,<0.19.0"
```

### Conflict 3: PyTorch Index URL
**Issue**: Multiple index URLs can cause confusion

**Resolution**:
```bash
# Install PyTorch FIRST with correct index
pip install torch torchvision torchaudio --index-url <correct-url>
# Then install other packages normally
pip install transformers sentence-transformers
```

### Conflict 4: OpenCV Multiple Versions
**Issue**: opencv-python and opencv-python-headless conflict

**Resolution**:
```bash
# Uninstall all versions
pip uninstall opencv-python opencv-python-headless opencv-contrib-python
# Install only one version
pip install opencv-python==4.8.1.78
```

## 🔄 Alternative Versions

### For Different Python Versions

#### Python 3.7 (Legacy)
```
torch==1.13.1
transformers==4.26.0
sentence-transformers==2.2.0
numpy==1.21.6
pandas==1.3.5
scikit-learn==1.0.2
```

#### Python 3.12 (Experimental)
```
torch==2.2.0
transformers==4.38.0
sentence-transformers==2.5.0
numpy==1.26.0
pandas==2.2.0
scikit-learn==1.4.0
```

### For Different CUDA Versions

#### CUDA 11.7
```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117
```

#### CUDA 12.1
```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

### Lighter Alternatives

For systems with limited resources:

```
# Minimal installation (CPU only, no image models)
pip install numpy pandas scikit-learn lightgbm
pip install transformers sentence-transformers
pip install Pillow requests tqdm
```

## 🧪 Testing Matrix

### Tested Combinations

| Config | Python | PyTorch | CUDA | OS | Status |
|--------|--------|---------|------|----|----|
| A | 3.9 | 2.1.2 | CPU | Windows 11 | ✅ |
| B | 3.10 | 2.1.2 | 11.8 | Ubuntu 22.04 | ✅ |
| C | 3.11 | 2.1.2 | CPU | macOS 13 | ✅ |
| D | 3.8 | 2.1.2 | 11.8 | Windows 10 | ✅ |
| E | 3.10 | 2.1.2 | 12.1 | Ubuntu 20.04 | ✅ |

## 📋 Installation Order

**Critical**: Install in this order to avoid conflicts:

1. **Base packages**
   ```bash
   pip install numpy scipy
   ```

2. **PyTorch** (with correct index)
   ```bash
   pip install torch torchvision torchaudio --index-url <url>
   ```

3. **Transformers stack**
   ```bash
   pip install transformers tokenizers huggingface-hub safetensors
   ```

4. **Sentence Transformers**
   ```bash
   pip install sentence-transformers
   ```

5. **ML packages**
   ```bash
   pip install scikit-learn lightgbm
   ```

6. **Data science**
   ```bash
   pip install pandas matplotlib seaborn
   ```

7. **Image processing**
   ```bash
   pip install Pillow opencv-python
   ```

8. **Utilities**
   ```bash
   pip install requests tqdm
   ```

## 🔧 Verification Commands

### Check Installed Versions
```python
import torch
import transformers
import sentence_transformers
import lightgbm
import numpy
import pandas

print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"Sentence-Transformers: {sentence_transformers.__version__}")
print(f"LightGBM: {lightgbm.__version__}")
print(f"NumPy: {numpy.__version__}")
print(f"Pandas: {pandas.__version__}")
```

### Check CUDA
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
```

### Run Full Verification
```bash
python verify_setup.py
```

## 📞 Support Resources

- **PyTorch Installation**: https://pytorch.org/get-started/locally/
- **Transformers**: https://huggingface.co/docs/transformers/installation
- **LightGBM**: https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html
- **Issue Tracker**: Check project issues on GitHub

## 🗓️ Update Schedule

- **Monthly**: Security patches
- **Quarterly**: Minor version updates
- **Annually**: Major version updates

**Last Verified**: October 11, 2025
**Next Review**: January 11, 2026

---

**Note**: These versions are specifically tested for the Amazon ML Challenge project. For other projects, versions may differ.
