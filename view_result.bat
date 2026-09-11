@echo off
REM View a MASt3R-SLAM reconstruction (point cloud + camera trajectory) in
REM an interactive open3d window.
REM Usage:
REM   view_result.bat logs\<run>\<seq>.ply logs\<run>\<seq>.txt [--frustum-stride N]

if "%~2"=="" (
  echo Usage: view_result.bat ^<path\to\seq.ply^> ^<path\to\seq.txt^> [--frustum-stride N]
  echo   e.g. view_result.bat logs\cyberpunk_test\rgb.ply logs\cyberpunk_test\rgb.txt --frustum-stride 5
  exit /b 1
)

cd /d C:\workspace\NAVIGATION\MASt3R-SLAM
.venv\Scripts\python.exe view_result.py %*
