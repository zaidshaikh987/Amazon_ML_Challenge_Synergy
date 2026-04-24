@echo off
setlocal enabledelayedexpansion

echo.
echo ========================================================================
echo Amazon ML Challenge - Automated Installation
echo ========================================================================
echo.

REM Step 1: Create virtual environment
echo [Step 1/5] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists
) else (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        exit /b 1
    )
    echo SUCCESS: Virtual environment created
)

REM Step 2: Activate and upgrade pip
echo.
echo [Step 2/5] Upgrading pip, setuptools, wheel...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel --quiet
echo SUCCESS: pip upgraded

REM Step 3: Install PyTorch
echo.
echo [Step 3/5] Installing PyTorch 2.5.1 (CPU)...
echo This may take 2-3 minutes...
pip install torch==2.5.1+cpu torchvision==0.20.1+cpu torchaudio==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch
    exit /b 1
)
echo SUCCESS: PyTorch installed

REM Step 4: Install all other packages
echo.
echo [Step 4/5] Installing remaining packages...

echo   ^> Installing core packages...
pip install numpy==1.24.3 pandas==2.0.3 scipy==1.11.4

echo   ^> Installing ML packages...
pip install scikit-learn==1.3.2 lightgbm==4.1.0

echo   ^> Installing transformers...
pip install transformers==4.36.2 tokenizers==0.15.0 huggingface-hub==0.20.1 safetensors==0.4.1

echo   ^> Installing sentence-transformers...
pip install sentence-transformers==2.2.2

echo   ^> Installing image processing...
pip install Pillow==10.1.0 opencv-python==4.8.1.78

echo   ^> Installing utilities...
pip install requests==2.31.0 urllib3==2.1.0 certifi==2023.11.17 tqdm==4.66.1

echo   ^> Installing visualization...
pip install matplotlib==3.8.2 seaborn==0.13.0

echo   ^> Installing additional dependencies...
pip install regex==2023.12.25 filelock==3.13.1 packaging==23.2 pyyaml==6.0.1 typing-extensions==4.9.0 sympy==1.12 joblib==1.3.2 threadpoolctl==3.2.0

echo.
echo SUCCESS: All packages installed

REM Step 5: Verify installation
echo.
echo [Step 5/5] Verifying installation...
python verify_setup.py

echo.
echo ========================================================================
echo Installation Complete!
echo ========================================================================
echo.
echo Virtual environment: %CD%\venv
echo.
echo To activate: venv\Scripts\activate.bat
echo To run project: run_sample.bat
echo.
