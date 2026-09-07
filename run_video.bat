@echo off
REM Run MASt3R-SLAM on an arbitrary video file or image folder (no curated
REM dataset needed) -- uncalibrated mono processing via config/base.yaml.
REM Uses the venv's python.exe by full path -- this venv's pip.exe/activate
REM launcher stubs were broken by an earlier folder move, bare "python" on
REM PATH is not reliable here.

if "%~1"=="" (
  echo Usage: run_video.bat ^<path\to\video.mp4 or image folder^> [config]
  echo   e.g. run_video.bat C:\videos\outdoor_walk.mp4
  echo   e.g. run_video.bat C:\videos\outdoor_walk.mp4 config\base.yaml
  exit /b 1
)

set VIDEO=%~1
set CONFIG=%~2
if "%CONFIG%"=="" set CONFIG=config\base.yaml

cd /d C:\workspace\NAVIGATION\MASt3R-SLAM
C:\workspace\NAVIGATION\MASt3R-SLAM\.venv\Scripts\python.exe main.py --dataset "%VIDEO%" --config "%CONFIG%"
