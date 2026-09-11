@echo off
REM Run MASt3R-SLAM on TUM freiburg1_room and compare against its own
REM (real, TUM-native) groundtruth.txt via evo -- no isekai-style conversion
REM needed since TUM sequences ship groundtruth.txt already.

cd /d C:\workspace\NAVIGATION\MASt3R-SLAM

set SEQ_DIR=datasets\tum\rgbd_dataset_freiburg1_room
set TGZ=datasets\tum\rgbd_dataset_freiburg1_room.tgz
set URL=https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz

if not exist "%SEQ_DIR%\groundtruth.txt" (
  echo [run_room_eval] groundtruth.txt missing -- local .tgz is likely truncated, re-downloading...
  if exist "%TGZ%" del "%TGZ%"
  if exist "%SEQ_DIR%" rmdir /s /q "%SEQ_DIR%"
  curl -L --progress-bar -o "%TGZ%" "%URL%"
  if errorlevel 1 (
    echo ERROR: download failed.
    exit /b 1
  )
  mkdir "%SEQ_DIR%"
  tar -xzf "%TGZ%" -C datasets\tum
)

if not exist "%SEQ_DIR%\groundtruth.txt" (
  echo ERROR: groundtruth.txt still missing after re-download, aborting.
  exit /b 1
)

echo [run_room_eval] running MASt3R-SLAM on freiburg1_room...
.venv\Scripts\python.exe main.py --dataset "%SEQ_DIR%\" --config config\calib.yaml --no-viz --save-as freiburg1_room_eval
if errorlevel 1 (
  echo ERROR: main.py failed, see output above.
  exit /b 1
)

set EST=logs\freiburg1_room_eval\rgbd_dataset_freiburg1_room.txt
if not exist "%EST%" (
  echo ERROR: expected output %EST% not found -- check logs\freiburg1_room_eval\ for the actual filename.
  exit /b 1
)

echo [run_room_eval] APE (Sim(3) aligned, scale-corrected):
.venv\Scripts\evo_ape.exe tum "%SEQ_DIR%\groundtruth.txt" "%EST%" -a -s

echo [run_room_eval] plotting trajectories...
.venv\Scripts\evo_traj.exe tum "%SEQ_DIR%\groundtruth.txt" "%EST%" --ref "%SEQ_DIR%\groundtruth.txt" -a -s --sync -p
