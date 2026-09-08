@echo off
setlocal enabledelayedexpansion
REM Full rebuild of MASt3R-SLAM against the new Python 3.11 venv (.venv),
REM encoding every fix found during tonight's Python 3.12 build saga.
REM MUST be run from the "x64 Native Tools Command Prompt for VS 2022"
REM (Start Menu) -- plain PowerShell/cmd will fail with WinError 2 (no cl.exe).

cd /d C:\workspace\NAVIGATION\MASt3R-SLAM
where cl >nul 2>&1
if errorlevel 1 (
  echo ERROR: cl.exe not found. Run this from "x64 Native Tools Command Prompt for VS 2022".
  exit /b 1
)

set PY=C:\workspace\NAVIGATION\MASt3R-SLAM\.venv\Scripts\python.exe
set TORCH_CUDA_ARCH_LIST=12.0
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set PIP_EXTRA_INDEX_URL=https://pypi.org/simple

echo [1/11] Upgrading pip/setuptools/wheel...
"%PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :fail

echo [2/11] Installing torch/torchvision/torchaudio 2.7.0+cu128...
"%PY%" -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :fail

echo [3/11] Installing Cython<0.30 (pyimgui needs this exact range, NOT latest 3.x)...
"%PY%" -m pip install "Cython<0.30"
if errorlevel 1 goto :fail

echo [4/11] Building lietorch from source (TORCH_CUDA_ARCH_LIST=12.0 for Blackwell)...
"%PY%" -m pip install --no-build-isolation --no-deps "lietorch @ git+https://github.com/princeton-vl/lietorch.git"
if errorlevel 1 goto :fail

echo [5/11] Installing MASt3R (--no-deps, skips curope which has a working CPU fallback)...
"%PY%" -m pip install -e thirdparty/mast3r --no-build-isolation --no-deps
if errorlevel 1 goto :fail

echo [6/11] Installing asmk from its own subpath (builds fine standalone)...
if exist thirdparty\mast3r\asmk\build rmdir /s /q thirdparty\mast3r\asmk\build
if exist thirdparty\mast3r\asmk\asmk.egg-info rmdir /s /q thirdparty\mast3r\asmk\asmk.egg-info
"%PY%" -m pip install thirdparty/mast3r/asmk --no-build-isolation
if errorlevel 1 goto :fail

echo [7/11] Installing faiss-cpu (not declared anywhere in this repo's own deps)...
"%PY%" -m pip install faiss-cpu scikit-learn
if errorlevel 1 goto :fail

echo [8/12] Ensuring nested pyimgui submodule (imgui-cpp) is initialized...
pushd thirdparty\in3d\thirdparty\pyimgui
git submodule update --init --recursive
popd
REM Clean stale build artifacts from any earlier attempt -- pyimgui's build/
REM dir persists across venv rebuilds (it's a real git checkout, not inside
REM .venv) and causes WinError 183 "file already exists" if left in place.
if exist thirdparty\in3d\thirdparty\pyimgui\build rmdir /s /q thirdparty\in3d\thirdparty\pyimgui\build
if exist thirdparty\in3d\thirdparty\pyimgui\imgui.egg-info rmdir /s /q thirdparty\in3d\thirdparty\pyimgui\imgui.egg-info

echo [9/12] Installing in3d (imgui/moderngl/moderngl-window)...
"%PY%" -m pip install -e thirdparty/in3d --no-build-isolation
if errorlevel 1 goto :fail

echo [10/12] Installing MASt3R-SLAM itself (mast3r_slam_backends CUDA extension)...
"%PY%" -m pip install --no-build-isolation -e .
if errorlevel 1 goto :fail

echo [11/12] Installing opencv-python (mast3r's own dep, skipped by --no-deps
echo          earlier -- pinned ^<5 since 5.x forces numpy^>=2, which breaks
echo          mast3r_slam/dataloader.py's np.unicode_ usage, removed in numpy 2.0)...
"%PY%" -m pip install "opencv-python<5" "numpy<2"
if errorlevel 1 goto :fail

echo [12/12] Verifying everything imports...
"%PY%" -c "import mast3r; import mast3r_slam; import mast3r_slam_backends; import lietorch; import asmk; import faiss; import imgui; import moderngl; import cv2; import numpy; print('ALL OK', numpy.__version__)"
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo SUCCESS. Run with:
echo   C:\workspace\NAVIGATION\MASt3R-SLAM\.venv\Scripts\python.exe main.py --dataset datasets/tum/rgbd_dataset_freiburg1_360/ --config config/calib.yaml
echo ============================================================
exit /b 0

:fail
echo.
echo ============================================================
echo FAILED at the step above. Check the output for the real error.
echo ============================================================
exit /b 1
