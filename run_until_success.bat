@echo off
REM Retry wrapper for the intermittent gauss_newton_rays CUDA crash (see
REM SETUP_TROUBLESHOOTING.md item 18) -- keeps re-running the same command
REM until main.py exits 0 (clean "done"), or MAX_TRIES is hit.
REM Usage: run_until_success.bat <dataset> <config> <save-as> [extra args...]

if "%~1"=="" (
  echo Usage: run_until_success.bat ^<dataset^> ^<config^> ^<save-as^> [extra args...]
  echo   e.g. run_until_success.bat "C:\path\to\video.mp4" config\base.yaml my_run --skip-frames 100
  exit /b 1
)

cd /d C:\workspace\NAVIGATION\MASt3R-SLAM

set DATASET=%~1
set CONFIG=%~2
set SAVEAS=%~3
shift
shift
shift
set EXTRA=%1 %2 %3 %4 %5 %6 %7 %8 %9

set MAX_TRIES=8
set /a TRY=0

:retry
set /a TRY+=1
echo [run_until_success] attempt %TRY% / %MAX_TRIES%
.venv\Scripts\python.exe main.py --dataset "%DATASET%" --config "%CONFIG%" --no-viz --save-as "%SAVEAS%" %EXTRA%
if %ERRORLEVEL% EQU 0 (
  echo [run_until_success] succeeded on attempt %TRY%
  exit /b 0
)

echo [run_until_success] attempt %TRY% failed with exit code %ERRORLEVEL%
if %TRY% GEQ %MAX_TRIES% (
  echo [run_until_success] giving up after %MAX_TRIES% attempts.
  exit /b 1
)
goto retry
