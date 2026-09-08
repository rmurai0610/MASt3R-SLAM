@echo off
REM Run MASt3R-SLAM on one of the longer walkthrough-style TUM sequences
REM (downloaded via download_tum_walkthrough.bat).
REM Usage:
REM   run_walkthrough.bat                 -> freiburg3_long_office_household (default)
REM   set SEQUENCE=freiburg2_large_with_loop & run_walkthrough.bat

cd /d C:\workspace\NAVIGATION\MASt3R-SLAM

if not defined SEQUENCE set SEQUENCE=rgbd_dataset_freiburg3_long_office_household
if not defined CONFIG set CONFIG=config\calib.yaml

set DATASET_DIR=datasets\tum\%SEQUENCE%
if not exist "%DATASET_DIR%\rgb.txt" (
  echo ERROR: %DATASET_DIR%\rgb.txt not found.
  echo   Run download_tum_walkthrough.bat first.
  exit /b 1
)

echo [run_walkthrough] sequence=%SEQUENCE% config=%CONFIG%
C:\workspace\NAVIGATION\MASt3R-SLAM\.venv\Scripts\python.exe main.py --dataset "%DATASET_DIR%\" --config "%CONFIG%"
