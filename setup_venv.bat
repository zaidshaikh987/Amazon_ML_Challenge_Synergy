@echo off
REM ========================================
REM Amazon ML Challenge - Virtual Environment Setup
REM ========================================

echo.
echo ============================================
echo Amazon ML Challenge - Environment Setup
echo ============================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [INFO] Python found:
python --version
echo.

REM Ask user for GPU support
set /p GPU_SUPPORT="Do you have NVIDIA GPU and want CUDA support? (y/n): "

if /i "%GPU_SUPPORT%"=="y" (
    echo [INFO] Will install PyTorch with CUDA 11.8 support
    set REQ_FILE=requirements-gpu.txt
) else (
    echo [INFO] Will install CPU-only version
    set REQ_FILE=requirements.txt
)
echo.

REM Remove old virtual environment if exists
if exist venv (
    echo [INFO] Removing existing virtual environment...
    rmdir /s /q venv
)

echo [INFO] Creating new virtual environment...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment!
    pause
    exit /b 1
)

echo [SUCCESS] Virtual environment created!
echo.

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b 1
)

echo [INFO] Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip!
    pause
    exit /b 1
)
echo.

echo [INFO] Installing packages from %REQ_FILE%...
echo This may take several minutes...
echo.

REM Install PyTorch first
if /i "%GPU_SUPPORT%"=="y" (
    echo [INFO] Installing PyTorch 2.5.1 with CUDA 11.8...
    pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118
) else (
    echo [INFO] Installing PyTorch 2.5.1 (CPU only)...
    pip install torch==2.5.1+cpu torchvision==0.20.1+cpu torchaudio==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
)

if errorlevel 1 (
    echo [ERROR] Failed to install PyTorch!
    pause
    exit /b 1
)
echo.

echo [INFO] Installing remaining packages...
pip install numpy==1.24.3 pandas==2.0.3 scipy==1.11.4
pip install scikit-learn==1.3.2 lightgbm==4.1.0
pip install transformers==4.36.2 sentence-transformers==2.2.2
pip install tokenizers==0.15.0 huggingface-hub==0.20.1 safetensors==0.4.1
pip install Pillow==10.1.0 opencv-python==4.8.1.78
pip install requests==2.31.0 urllib3==2.1.0 certifi==2023.11.17 tqdm==4.66.1
pip install matplotlib==3.8.2 seaborn==0.13.0
pip install regex==2023.12.25 filelock==3.13.1 packaging==23.2 pyyaml==6.0.1
pip install typing-extensions==4.9.0 sympy==1.12 joblib==1.3.2 threadpoolctl==3.2.0

if errorlevel 1 (
    echo [ERROR] Failed to install packages!
    pause
    exit /b 1
)
echo.

echo ============================================
echo [SUCCESS] Installation Complete!
echo ============================================
echo.
echo Virtual environment is ready at: %CD%\venv
echo.
echo To activate the environment, run:
echo     venv\Scripts\activate.bat
echo.
echo To deactivate, run:
echo     deactivate
echo.

REM Verify installation
echo [INFO] Verifying installation...
echo.
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import sentence_transformers; print(f'Sentence-Transformers: {sentence_transformers.__version__}')"
python -c "import lightgbm; print(f'LightGBM: {lightgbm.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"

if /i "%GPU_SUPPORT%"=="y" (
    echo.
    echo [INFO] Checking CUDA availability...
    python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
)

echo.
echo ============================================
echo Setup complete! You can now run the project.
echo ============================================
echo.
pause
