@echo off
setlocal enabledelayedexpansion
REM Download the TUM RGB-D room sequence (if not already present) and run
REM MASt3R-SLAM on it. Uses the venv's python.exe by full path directly --
REM this venv's pip.exe/activate launcher stubs were broken by an earlier
REM folder move, so bare "python"/"pip" on PATH is not reliable here.

cd /d C:\workspace\NAVIGATION\MASt3R-SLAM

set PY=C:\workspace\NAVIGATION\MASt3R-SLAM\.venv\Scripts\python.exe
if not exist "%PY%" (
  echo ERROR: venv python not found at %PY%
  echo   Run setup.bat first.
  exit /b 1
)

if not defined SEQUENCE set SEQUENCE=rgbd_dataset_freiburg1_room
if not defined CONFIG set CONFIG=config\calib.yaml

set DATASET_DIR=datasets\tum\%SEQUENCE%
if not exist "%DATASET_DIR%\rgb.txt" (
  echo [run_tum_demo] %DATASET_DIR% missing or incomplete -- downloading TUM dataset...
  "%PY%" download_datasets.py --tum
  if errorlevel 1 (
    echo ERROR: dataset download failed.
    exit /b 1
  )
)

if not exist "%DATASET_DIR%\rgb.txt" (
  echo ERROR: %DATASET_DIR%\rgb.txt still missing after download.
  echo   Check download_datasets.py output above, or pick a different SEQUENCE=.
  exit /b 1
)

echo [run_tum_demo] sequence=%SEQUENCE% config=%CONFIG%
"%PY%" main.py --dataset "%DATASET_DIR%\" --config "%CONFIG%"
