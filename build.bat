@echo off
setlocal enabledelayedexpansion

echo MASt3R-SLAM Build Script
echo ========================
echo.

if not exist "mast3r_slam" (
    echo Error: Run this script from the MASt3R-SLAM root directory
    exit /b 1
)

echo Checking prerequisites...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Install Python or activate the conda environment.
    exit /b 1
)

echo Python found:
python --version

echo.
echo Checking CUDA availability...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo Warning: CUDA toolkit not found. Build may fail if CUDA is not installed.
    echo Make sure to have nvcc in PATH or use WSL.
) else (
    echo NVIDIA GPU detected:
    nvidia-smi --query-gpu=name --format=csv,noheader
)

echo.
echo Step 1: Building MASt3R-SLAM with CUDA support...
echo This will compile C++ and CUDA extensions.
echo.

pip install --no-build-isolation -e .
if errorlevel 1 (
    echo Error: Build failed. Check the output above for details.
    echo Common issues:
    echo   - CUDA toolkit not in PATH: run vcvarsall.bat or add CUDA to PATH
    echo   - PyTorch not installed or wrong CUDA version: run setup.bat
    echo   - Missing build tools on Windows: install Visual Studio Build Tools
    exit /b 1
)

echo.
echo Build completed successfully!
echo MASt3R-SLAM backend extensions compiled.
echo.
echo Next steps:
echo   1. Download models: get_models.bat
echo   2. Download example datasets: examples.bat
echo   3. Run demo: python main.py --dataset ^<path^> --config config/base.yaml
echo.
