@echo off
setlocal enabledelayedexpansion

echo MASt3R-SLAM Examples and Datasets
echo ==================================
echo.

if not exist "mast3r_slam" (
    echo Error: Run this script from the MASt3R-SLAM root directory
    exit /b 1
)

echo Checking prerequisites...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Activate the conda environment first.
    exit /b 1
)

if not exist "checkpoints\MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" (
    echo Error: Models not found. Run get_models.bat first.
    exit /b 1
)

echo.
echo Available examples:
echo ==================
echo.
echo 1. TUM-RGBD Dataset (small, ~1-2 GB)
echo    - Recommended for first-time users
echo    - Download and run: download_tum_examples.bat
echo    - Run: python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_room/ --config config/calib.yaml
echo.
echo 2. 7-Scenes Dataset (medium, ~10 GB)
echo    - Download and run: download_7_scenes_examples.bat
echo    - Run: python main.py --dataset datasets/7_scenes/^<scene^>/ --config config/calib.yaml
echo.
echo 3. EuRoC Dataset (medium, ~15 GB)
echo    - Download and run: download_euroc_examples.bat
echo    - Run: python main.py --dataset datasets/euroc/^<sequence^>/ --config config/base.yaml
echo.
echo 4. ETH3D SLAM Dataset (large, ~50 GB)
echo    - Download and run: download_eth3d_examples.bat
echo    - Run: python main.py --dataset datasets/eth3d/^<sequence^>/ --config config/eth3d.yaml
echo.
echo 5. Live RealSense Camera (requires camera)
echo    - Run: python main.py --dataset realsense --config config/base.yaml
echo.
echo 6. Video or Image Folder
echo    - Run: python main.py --dataset ^<path/to/video.mp4^> --config config/base.yaml
echo    - Run: python main.py --dataset ^<path/to/images/^> --config config/base.yaml
echo.
echo Example with camera intrinsics:
echo    - Run: python main.py --dataset ^<path^> --config config/base.yaml --calib config/intrinsics.yaml
echo.
echo ==================
echo.
echo Download scripts explanation:
echo    - TUM: Small dataset, fast download (~10 min on fast internet)
echo    - 7-Scenes: Multiple indoor scenes (~1-2 hours download)
echo    - EuRoC: Drone sequences with ground truth (~1 hour download)
echo    - ETH3D: High-quality SLAM sequences (~2-3 hours download)
echo.
echo === Quick Start ===
echo Run this command to download and test TUM dataset:
echo.
echo   cd /d "!CD!" ^& bash scripts/download_tum.sh ^& python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_room/ --config config/calib.yaml
echo.
echo Note: On Windows, bash scripts require Git Bash or WSL. Alternative:
echo   Use the evaluation scripts: eval_tum.sh, eval_7_scenes.sh, etc.
echo.
