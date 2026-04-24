#!/usr/bin/env python3
"""
Verify that all required packages are installed and working correctly.
"""

import sys
import importlib.util

def check_package(package_name, import_name=None, min_version=None):
    """Check if a package is installed and optionally verify version."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        
        if min_version and version != 'unknown':
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                print(f"❌ {package_name}: {version} (requires {min_version}+)")
                return False
        
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: NOT INSTALLED")
        return False
    except Exception as e:
        print(f"⚠️  {package_name}: ERROR - {str(e)}")
        return False

def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA Available: Yes")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   Device Count: {torch.cuda.device_count()}")
            print(f"   Device Name: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("ℹ️  CUDA Available: No (CPU only)")
            return False
    except Exception as e:
        print(f"⚠️  CUDA Check: ERROR - {str(e)}")
        return False

def test_imports():
    """Test critical imports for the project."""
    print("\n🔍 Testing Critical Imports...")
    print("=" * 60)
    
    tests_passed = True
    
    # Test PyTorch
    try:
        import torch
        print("✅ PyTorch import: OK")
        
        # Test basic tensor operations
        x = torch.rand(3, 3)
        y = x * 2
        print("✅ PyTorch operations: OK")
    except Exception as e:
        print(f"❌ PyTorch test: FAILED - {str(e)}")
        tests_passed = False
    
    # Test Transformers
    try:
        from transformers import AutoTokenizer
        print("✅ Transformers import: OK")
    except Exception as e:
        print(f"❌ Transformers test: FAILED - {str(e)}")
        tests_passed = False
    
    # Test Sentence Transformers
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ Sentence-Transformers import: OK")
    except Exception as e:
        print(f"❌ Sentence-Transformers test: FAILED - {str(e)}")
        tests_passed = False
    
    # Test LightGBM
    try:
        import lightgbm as lgb
        print("✅ LightGBM import: OK")
    except Exception as e:
        print(f"❌ LightGBM test: FAILED - {str(e)}")
        tests_passed = False
    
    # Test Image Processing
    try:
        from PIL import Image
        import cv2
        print("✅ Image processing (PIL, OpenCV): OK")
    except Exception as e:
        print(f"❌ Image processing test: FAILED - {str(e)}")
        tests_passed = False
    
    # Test Data Science
    try:
        import numpy as np
        import pandas as pd
        import sklearn
        print("✅ Data science stack (NumPy, Pandas, Sklearn): OK")
    except Exception as e:
        print(f"❌ Data science test: FAILED - {str(e)}")
        tests_passed = False
    
    return tests_passed

def main():
    """Main verification routine."""
    print("\n" + "=" * 60)
    print("🔧 Amazon ML Challenge - Environment Verification")
    print("=" * 60)
    
    # Check Python version
    print(f"\n📌 Python Version: {sys.version}")
    py_version = sys.version_info
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 8):
        print("❌ Python 3.8+ is required!")
        sys.exit(1)
    else:
        print("✅ Python version OK")
    
    # Check packages
    print("\n📦 Checking Package Installations...")
    print("=" * 60)
    
    all_ok = True
    
    # Core packages
    all_ok &= check_package("numpy", min_version="1.21.0")
    all_ok &= check_package("pandas", min_version="1.5.0")
    all_ok &= check_package("scipy", min_version="1.7.0")
    
    # ML packages
    all_ok &= check_package("scikit-learn", "sklearn", min_version="1.1.0")
    all_ok &= check_package("lightgbm", min_version="3.3.0")
    
    # PyTorch
    all_ok &= check_package("torch", min_version="1.12.0")
    all_ok &= check_package("torchvision", min_version="0.13.0")
    
    # Transformers
    all_ok &= check_package("transformers", min_version="4.20.0")
    all_ok &= check_package("sentence-transformers", "sentence_transformers", min_version="2.2.0")
    all_ok &= check_package("tokenizers", min_version="0.13.0")
    
    # Image processing
    all_ok &= check_package("Pillow", "PIL", min_version="9.0.0")
    all_ok &= check_package("opencv-python", "cv2")
    
    # Utilities
    all_ok &= check_package("requests", min_version="2.28.0")
    all_ok &= check_package("tqdm", min_version="4.64.0")
    all_ok &= check_package("matplotlib", min_version="3.5.0")
    all_ok &= check_package("seaborn", min_version="0.11.0")
    
    # Check CUDA
    print("\n🎮 GPU/CUDA Information...")
    print("=" * 60)
    check_cuda()
    
    # Test imports
    tests_passed = test_imports()
    
    # Final summary
    print("\n" + "=" * 60)
    if all_ok and tests_passed:
        print("✅ All checks passed! Environment is ready.")
        print("=" * 60)
        print("\n🚀 You can now run the project:")
        print("   python src/extract_embeddings.py --help")
        print("   python src/train_baseline.py --help")
        print("   python src/predict.py --help")
        print()
        return 0
    else:
        print("❌ Some checks failed. Please review errors above.")
        print("=" * 60)
        print("\n🔧 Try:")
        print("   1. Reinstall failed packages")
        print("   2. Check ENVIRONMENT_SETUP.md for troubleshooting")
        print("   3. Run setup script again: setup_venv.bat (Windows) or ./setup_venv.sh (Linux/Mac)")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
