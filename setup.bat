@echo off
setlocal enabledelayedexpansion

echo MASt3R-SLAM Setup Script
echo ========================
echo.
echo This script uses Python venv (no conda required).
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
    echo Download from: https://www.python.org/downloads/
    exit /b 1
)

echo.
echo Step 1: Creating virtual environment...
if exist ".venv" (
    echo Virtual environment already exists at .venv
) else (
    python -m venv .venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment
        exit /b 1
    )
    echo Virtual environment created
)

echo.
echo Step 2: Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    exit /b 1
)

echo.
echo Step 3: Checking CUDA availability...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo Warning: CUDA not detected. Make sure CUDA Toolkit is installed.
) else (
    echo NVIDIA GPU detected:
    nvidia-smi --query-gpu=name --format=csv,noheader
)

echo.
echo Step 4: Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo Error: Failed to upgrade build tools
    exit /b 1
)

echo.
echo Step 5: Installing PyTorch with CUDA support...
REM Default to CUDA 12.8. To change, edit this line or set cuda_choice before running.
set cuda_choice=12.8

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
    echo Error: Invalid CUDA version. Please edit setup.bat to change cuda_choice.
    exit /b 1
)

echo Installing PyTorch 2.7.0 with CUDA !cuda_ver!...
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/!cuda_index!
if errorlevel 1 (
    echo Error: Failed to install PyTorch
    exit /b 1
)

echo.
echo Step 6: Verifying PyTorch installation...
python -c "import torch; print(f'PyTorch {torch.__version__} OK'); print(f'CUDA {torch.version.cuda}')"
if errorlevel 1 (
    echo Error: PyTorch verification failed
    exit /b 1
)

echo.
echo Step 7: Checking for MSVC C++ compiler (needed to build curope/asmk/lietorch)...
where cl >nul 2>&1
if errorlevel 1 (
    set VSWHERE=C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe
    set VSINSTALLPATH=
    if exist "!VSWHERE!" (
        for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VSINSTALLPATH=%%i
    )
    if defined VSINSTALLPATH (
        echo cl.exe not on PATH -- loading MSVC environment from "!VSINSTALLPATH!"...
        call "!VSINSTALLPATH!\VC\Auxiliary\Build\vcvarsall.bat" x64
        where cl >nul 2>&1
    )
    if errorlevel 1 (
        echo Error: MSVC C++ build tools not found or not usable.
        echo   curope/asmk ^(MASt3R's CUDA extensions^) and lietorch require cl.exe to build.
        echo   Fix, from an ELEVATED ^(Run as Administrator^) PowerShell:
        echo     ^& "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vs_installer.exe" modify --installPath "C:\Program Files (x86)\Microsoft Visual Studio\2022\Community" --add Microsoft.VisualStudio.Workload.NativeDesktop --includeRecommended --passive --norestart
        echo   Then close this terminal, open a new one, and re-run setup.bat.
        exit /b 1
    )
    echo MSVC environment loaded.
) else (
    echo cl.exe found on PATH.
)

echo.
echo Step 8: Installing MASt3R dependency...
pip install -e thirdparty/mast3r --no-build-isolation
if errorlevel 1 (
    echo Error: Failed to install MASt3R
    echo Troubleshooting: Make sure PyTorch is installed and torch can be imported
    python -c "import torch; print('PyTorch OK')" >nul 2>&1 || (
        echo ERROR: PyTorch not found. Verify step 5 completed successfully.
    )
    exit /b 1
)

echo.
echo Step 9: Installing in3d dependency...
pip install -e thirdparty/in3d --no-build-isolation
if errorlevel 1 (
    echo Error: Failed to install in3d
    exit /b 1
)

echo.
echo Step 10: Installing MASt3R-SLAM with CUDA support...
pip install --no-build-isolation -e .
if errorlevel 1 (
    echo Error: Failed to install MASt3R-SLAM
    echo Troubleshooting: Check that all dependencies installed successfully above
    exit /b 1
)

echo.
echo Step 11: Verifying installation...
python -c "import mast3r_slam; print('MASt3R-SLAM OK')"
if errorlevel 1 (
    echo Error: MASt3R-SLAM import failed
    exit /b 1
)

echo.
echo Step 12: Installing optional torchcodec for faster mp4 loading...
pip install torchcodec==0.1
if errorlevel 1 (
    echo Warning: Failed to install torchcodec (optional, continuing anyway)
)

echo.
echo ============================================
echo Setup completed successfully!
echo ============================================
echo.
echo Virtual environment: .venv
echo.
echo To activate the environment in future:
echo   .venv\Scripts\activate.bat
echo.
echo Next steps:
echo   1. Run get_models.bat to download checkpoints
echo   2. Run python download_datasets.py --tum to get sample data
echo   3. Run: python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_room/ --config config/calib.yaml
echo.
echo To deactivate the environment:
echo   deactivate
echo.
