@echo off
setlocal enabledelayedexpansion

echo MASt3R-SLAM Setup Script for Python venv
echo =========================================
echo.
echo This script uses Python venv instead of conda.
echo Make sure you have Python 3.11+ installed.
echo.

if not exist "mast3r_slam" (
    echo Error: Run this script from the MASt3R-SLAM root directory
    exit /b 1
)

echo Checking Python version...
python --version
if errorlevel 1 (
    echo Error: Python not found. Make sure Python 3.11+ is installed.
    exit /b 1
)

echo.
echo Step 1: Creating virtual environment...
if exist ".venv" (
    echo Virtual environment already exists at .venv
    set activate_script=.venv\Scripts\activate.bat
) else (
    python -m venv .venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment
        exit /b 1
    )
    set activate_script=.venv\Scripts\activate.bat
)

echo Activating virtual environment...
call !activate_script!
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    exit /b 1
)

echo.
echo Step 2: Checking CUDA availability...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo Warning: CUDA not detected. Building without GPU support may fail.
)

echo.
echo Step 3: Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo Error: Failed to upgrade build tools
    exit /b 1
)

echo.
echo Step 4: Installing PyTorch with CUDA support...
echo Choose your CUDA version:
echo   11.8: cu118
echo   12.1: cu121
echo   12.4: cu124
echo   12.8: cu128
echo.
set /p cuda_choice="Enter CUDA version (11.8, 12.1, 12.4, or 12.8): "

if "!cuda_choice!"=="11.8" (
    set cuda_index=cu118
    set cuda_ver=11.8
) else if "!cuda_choice!"=="12.1" (
    set cuda_index=cu121
    set cuda_ver=12.1
) else if "!cuda_choice!"=="12.4" (
    set cuda_index=cu124
    set cuda_ver=12.4
) else if "!cuda_choice!"=="12.8" (
    set cuda_index=cu128
    set cuda_ver=12.8
) else (
    echo Error: Invalid CUDA version. Please choose 11.8, 12.1, 12.4, or 12.8
    exit /b 1
)

echo Installing PyTorch 2.5.1 with CUDA !cuda_ver!...
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/!cuda_index!
if errorlevel 1 (
    echo Error: Failed to install PyTorch
    exit /b 1
)

echo.
echo Step 5: Verifying PyTorch installation...
python -c "import torch; print(f'PyTorch {torch.__version__} OK')"
if errorlevel 1 (
    echo Error: PyTorch verification failed
    exit /b 1
)

echo.
echo Step 6: Installing MASt3R dependency...
pip install -e thirdparty/mast3r --no-build-isolation
if errorlevel 1 (
    echo Error: Failed to install MASt3R
    exit /b 1
)

echo.
echo Step 7: Installing in3d dependency...
pip install -e thirdparty/in3d --no-build-isolation
if errorlevel 1 (
    echo Error: Failed to install in3d
    exit /b 1
)

echo.
echo Step 8: Installing MASt3R-SLAM with CUDA support...
pip install --no-build-isolation -e .
if errorlevel 1 (
    echo Error: Failed to install MASt3R-SLAM
    exit /b 1
)

echo.
echo Step 9: Verifying installation...
python -c "import mast3r_slam; print('MASt3R-SLAM OK')"
if errorlevel 1 (
    echo Error: MASt3R-SLAM import failed
    exit /b 1
)

echo.
echo Step 10: Installing optional torchcodec for faster mp4 loading...
pip install torchcodec==0.1
if errorlevel 1 (
    echo Warning: Failed to install torchcodec (optional, continuing anyway)
)

echo.
echo ============================================
echo Setup completed successfully!
echo ============================================
echo.
echo Virtual environment created at: .venv
echo.
echo To activate the environment in future:
echo   .venv\Scripts\activate.bat
echo.
echo Next steps:
echo   1. Run get_models.bat to download checkpoints
echo   2. Run examples.bat to download sample datasets
echo   3. Run "python main.py --dataset ^<path^> --config config/base.yaml"
echo.
echo To deactivate the environment:
echo   deactivate
echo.
