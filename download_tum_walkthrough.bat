@echo off
REM Download the longer walkthrough-style TUM sequences (freiburg2/3) using
REM curl directly, for real progress bars -- download_datasets.py's own
REM urllib.request.urlretrieve() has zero progress indication and looks
REM stalled on these multi-GB files even when working fine.

cd /d C:\workspace\NAVIGATION\MASt3R-SLAM
if not exist datasets\tum mkdir datasets\tum

echo [1/2] freiburg2_large_with_loop (~2.7 GB)...
curl -L --progress-bar -o datasets\tum\rgbd_dataset_freiburg2_large_with_loop.tgz "https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_large_with_loop.tgz"
if errorlevel 1 goto :fail

echo [2/2] freiburg3_long_office_household...
curl -L --progress-bar -o datasets\tum\rgbd_dataset_freiburg3_long_office_household.tgz "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz"
if errorlevel 1 goto :fail

echo [extracting] freiburg2_large_with_loop...
tar -xzf datasets\tum\rgbd_dataset_freiburg2_large_with_loop.tgz -C datasets\tum
echo [extracting] freiburg3_long_office_household...
tar -xzf datasets\tum\rgbd_dataset_freiburg3_long_office_household.tgz -C datasets\tum

echo.
echo Done. Run with:
echo   C:\workspace\NAVIGATION\MASt3R-SLAM\.venv\Scripts\python.exe main.py --dataset datasets/tum/rgbd_dataset_freiburg3_long_office_household/ --config config/calib.yaml
exit /b 0

:fail
echo.
echo FAILED -- check the curl output above.
exit /b 1
