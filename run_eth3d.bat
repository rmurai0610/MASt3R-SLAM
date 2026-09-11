@echo off
setlocal enabledelayedexpansion
REM Run MASt3R-SLAM on an ETH3D SLAM sequence (includes real outdoor scans).
REM ETH3D requires manual registration/download -- this script cannot fetch
REM it automatically. Uses the venv's python.exe by full path -- this venv's
REM pip.exe/activate launcher stubs were broken by an earlier folder move,
REM bare "python" on PATH is not reliable here.

if "%~1"=="" (
  echo Usage: run_eth3d.bat ^<sequence-name^>
  echo   e.g. run_eth3d.bat sofa_1
  echo.
  echo First time setup, if datasets\eth3d\^<sequence-name^> doesn't exist yet:
  echo   1. Register at https://www.eth3d.net/slam_datasets
  echo   2. Pick an outdoor sequence, download it
  echo   3. Extract it to: datasets\eth3d\^<sequence-name^>\
  exit /b 1
)

set SEQUENCE=%~1
set DATASET_DIR=datasets\eth3d\%SEQUENCE%

cd /d C:\workspace\NAVIGATION\MASt3R-SLAM

if not exist "%DATASET_DIR%" (
  echo ERROR: %DATASET_DIR% not found.
  echo.
  echo ETH3D requires manual registration/download -- this script cannot fetch it:
  echo   1. Register at https://www.eth3d.net/slam_datasets
  echo   2. Pick an outdoor sequence, download it
  echo   3. Extract it to: %DATASET_DIR%\
  exit /b 1
)

echo [run_eth3d] sequence=%SEQUENCE%
C:\workspace\NAVIGATION\MASt3R-SLAM\.venv\Scripts\python.exe main.py --dataset "%DATASET_DIR%\" --config config\eth3d.yaml
