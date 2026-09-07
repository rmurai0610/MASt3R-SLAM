@echo off
REM Run MASt3R-SLAM on the TUM freiburg1_room sequence.
REM Uses the venv's python.exe by full path -- this venv's pip.exe/activate
REM launcher stubs were broken by an earlier folder move, bare "python" on
REM PATH is not reliable here.

cd /d C:\workspace\NAVIGATION\MASt3R-SLAM
C:\workspace\NAVIGATION\MASt3R-SLAM\.venv\Scripts\python.exe main.py --dataset datasets/tum/rgbd_dataset_freiburg1_room/ --config config/calib.yaml
