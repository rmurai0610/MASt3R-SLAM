[comment]: <> (# MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors)

<p align="center">
  <h1 align="center">MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors</h1>
  <p align="center">
    <a href="https://rmurai.co.uk/"><strong>Riku Murai*</strong></a>
    ·
    <a href="https://edexheim.github.io/"><strong>Eric Dexheimer*</strong></a>
    ·
    <a href="https://www.doc.ic.ac.uk/~ajd/"><strong>Andrew J. Davison</strong></a>
  </p>
  <p align="center">(* Equal Contribution)</p>

[comment]: <> (  <h2 align="center">PAPER</h2>)
  <h3 align="center"><a href="https://arxiv.org/abs/2412.12392">Paper</a> | <a href="https://youtu.be/wozt71NBFTQ">Video</a> | <a href="https://edexheim.github.io/mast3r-slam/">Project Page</a></h3>
  <div align="center"></div>

<p align="center">
    <img src="./media/teaser.gif" alt="teaser" width="100%">
</p>
<br>

# Getting Started

## Quick Start with Docker (Recommended)

The easiest way to run MASt3R-SLAM is using Docker. This approach handles all dependencies automatically and produces results you can visualize immediately.

### 1. Build and Start Container
```bash
docker compose up -d --build
```

### 2. Run SLAM on Your Video
Place your video file in the `data/` directory and run:
```bash
# Use default (expects data/yourvideo.mp4)
bash ./scripts/run_docker.sh

# Or specify your video file
bash ./scripts/run_docker.sh data/myvideo.mov

# For low-memory systems (useful for large videos or limited GPU memory)
bash ./scripts/run_docker_low_memory.sh data/myvideo.mov
```

**Alternative**: Edit the script directly to change the default dataset path:
```bash
# Edit line 6 in scripts/run_docker.sh
DATASET=${1:-"data/your_actual_video.mp4"}
```

**Script Options:**
- `run_docker.sh` - Standard processing (recommended)
- `run_docker_low_memory.sh` - Optimized for systems with limited GPU memory (uses frame/image subsampling)

This will:
- Process your video using MASt3R-SLAM
- Save results to `./output/` directory:
  - `{video_name}.txt` - Camera trajectory (TUM format)
  - `{video_name}.ply` - 3D point cloud
  - `keyframes/` - Keyframe images with timestamps

### 3. Visualize Results
- **3D Point Cloud**: Open the `.ply` file in [CloudCompare](https://www.cloudcompare.org/) or [MeshLab](https://www.meshlab.net/)
- **Camera Trajectory**: Plot the `.txt` file using [evo toolkit](https://github.com/MichaelGrupp/evo):
  ```bash
  pip install evo
  evo_traj tum output/{video_name}.txt --plot --plot_mode xy
  ```
- **Keyframes**: View PNG images in `output/keyframes/{video_name}/`

### 4. Interactive Development (Optional)
For development or custom configurations, enter the container:
```bash
docker compose exec mast3r-slam bash
conda activate mast3r-slam
python main.py --dataset data/your_video.mp4 --config config/base.yaml
```

## Alternative: Local Installation

For users who prefer local installation or need custom modifications:
Check the system's CUDA version with nvcc
```
nvcc --version
```
Install pytorch with **matching** CUDA version following:
```
# CUDA 11.8
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# CUDA 12.4
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

Clone the repo and install the dependencies.
```
git clone https://github.com/rmurai0610/MASt3R-SLAM.git --recursive
cd MASt3R-SLAM/

# if you've clone the repo without --recursive run
# git submodule update --init --recursive

pip install -e thirdparty/mast3r
pip install -e thirdparty/in3d
pip install --no-build-isolation -e .
 

# Optionally install torchcodec for faster mp4 loading
pip install torchcodec==0.1
```

Setup the checkpoints for MASt3R and retrieval.  The license for the checkpoints and more information on the datasets used is written [here](https://github.com/naver/mast3r/blob/mast3r_sfm/CHECKPOINTS_NOTICE).
```
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/
```

### WSL Users
We have primarily tested on Ubuntu.  If you are using WSL, please checkout to the windows branch and follow the above installation.
```
git checkout windows
```
This disables multiprocessing which causes an issue with shared memory as discussed [here](https://github.com/rmurai0610/MASt3R-SLAM/issues/21).

## Advanced Usage

### Running with Dataset Examples
For standard datasets like TUM-RGBD:
```bash
# Download dataset
bash ./scripts/download_tum.sh

# Docker approach
docker compose exec mast3r-slam bash -c "conda activate mast3r-slam && python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_room/ --config config/calib.yaml"

# Local installation
python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_room/ --config config/calib.yaml
```

### Live Demo with Realsense Camera
```bash
# Docker approach
docker compose exec mast3r-slam bash -c "conda activate mast3r-slam && python main.py --dataset realsense --config config/base.yaml"

# Local installation
python main.py --dataset realsense --config config/base.yaml
```

### Custom Video/Image Processing
Our system can process MP4 videos or folders containing RGB images:

**Using Docker (saves results automatically):**
```bash
# Method 1: Use convenient script with argument
bash ./scripts/run_docker.sh data/your_video.mp4

# Method 2: Use low-memory version for large videos
bash ./scripts/run_docker_low_memory.sh data/your_video.mp4

# Method 3: Manual container execution
docker compose exec mast3r-slam bash -c "conda activate mast3r-slam && python main.py --dataset data/your_video.mp4 --config config/base.yaml --save-as output"
```

**Using Local Installation:**
```bash
python main.py --dataset <path/to/video>.mp4 --config config/base.yaml
python main.py --dataset <path/to/folder> --config config/base.yaml
```

**With Known Camera Calibration:**
If you have calibration parameters, specify them in `config/intrinsics.yaml`:
```bash
python main.py --dataset <path/to/video>.mp4 --config config/base.yaml --calib config/intrinsics.yaml
```

## Result Visualization

MASt3R-SLAM produces three main outputs that can be visualized:

### 1. 3D Point Cloud (`.ply` file)
The reconstructed 3D scene with colored points from all keyframes.

**Recommended viewers:**
- **[CloudCompare](https://www.cloudcompare.org/)** (Free, cross-platform)
  ```bash
  # Ubuntu/Debian
  sudo apt install cloudcompare
  # Then open: cloudcompare output/your_video.ply
  ```
- **[MeshLab](https://www.meshlab.net/)** (Free, cross-platform)
- **[Open3D](http://www.open3d.org/)** (Python library)
  ```python
  import open3d as o3d
  pcd = o3d.io.read_point_cloud("output/your_video.ply")
  o3d.visualization.draw_geometries([pcd])
  ```

### 2. Camera Trajectory (`.txt` file)
Camera poses in TUM RGBD format: `timestamp x y z qx qy qz qw`

**Visualization with evo toolkit:**
```bash
pip install evo
# 2D trajectory plot
evo_traj tum output/your_video.txt --plot --plot_mode xy
# 3D trajectory plot
evo_traj tum output/your_video.txt --plot --plot_mode xyz
# Save plot as image
evo_traj tum output/your_video.txt --plot --plot_mode xy --save_plot output/trajectory.png
```

**Custom Python visualization:**
```python
import numpy as np
import matplotlib.pyplot as plt

# Load trajectory
data = np.loadtxt('output/your_video.txt')
timestamps, positions = data[:, 0], data[:, 1:4]

# Plot 2D trajectory (top-down view)
plt.figure(figsize=(10, 8))
plt.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)
plt.scatter(positions[0, 0], positions[0, 1], color='green', s=100, label='Start')
plt.scatter(positions[-1, 0], positions[-1, 1], color='red', s=100, label='End')
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.title('Camera Trajectory (Top View)')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()
```

### 3. Keyframe Images
Selected keyframes with timestamps, saved as PNG files in `keyframes/{video_name}/`.

These can be viewed with any image viewer or used for further analysis:
```bash
# View keyframes with timestamps
ls -la output/keyframes/your_video/
# Example: 0.0.png, 10.2204.png, 20.4408.png...
```

### Real-time Visualization
For live visualization during processing, run without `--no-viz` flag:
```bash
# Docker
docker compose exec mast3r-slam bash -c "conda activate mast3r-slam && python main.py --dataset data/your_video.mp4 --config config/base.yaml"

# Local
python main.py --dataset your_video.mp4 --config config/base.yaml
```

This opens an interactive 3D viewer showing:
- Real-time point cloud reconstruction
- Camera trajectory with frustums
- Keyframe connections
- Adjustable confidence thresholds
- Multiple rendering modes (surfel/triangle)

## Downloading Dataset
### TUM-RGBD Dataset
```
bash ./scripts/download_tum.sh
```

### 7-Scenes Dataset
```
bash ./scripts/download_7_scenes.sh
```

### EuRoC Dataset
```
bash ./scripts/download_euroc.sh
```
### ETH3D SLAM Dataset
```
bash ./scripts/download_eth3d.sh
```

## Running Evaluations
All evaluation script will run our system in a single-threaded, headless mode.
We can run evaluations with/without calibration:
### TUM-RGBD Dataset
```
bash ./scripts/eval_tum.sh 
bash ./scripts/eval_tum.sh --no-calib
```

### 7-Scenes Dataset
```
bash ./scripts/eval_7_scenes.sh 
bash ./scripts/eval_7_scenes.sh --no-calib
```

### EuRoC Dataset
```
bash ./scripts/eval_euroc.sh 
bash ./scripts/eval_euroc.sh --no-calib
```
### ETH3D SLAM Dataset
```
bash ./scripts/eval_eth3d.sh 
```

## Reproducibility
There might be minor differences between the released version and the results in the paper after developing this multi-processing version. 
We run all our experiments on an RTX 4090, and the performance may differ when running with a different GPU.

## Acknowledgement
We sincerely thank the developers and contributors of the many open-source projects that our code is built upon.
- [MASt3R](https://github.com/naver/mast3r)
- [MASt3R-SfM](https://github.com/naver/mast3r/tree/mast3r_sfm)
- [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM)
- [ModernGL](https://github.com/moderngl/moderngl)

# Citation
If you found this code/work to be useful in your own research, please considering citing the following:

```bibtex
@article{murai2024_mast3rslam,
    title={{MASt3R-SLAM}: Real-Time Dense {SLAM} with {3D} Reconstruction Priors},
    author={Murai, Riku and Dexheimer, Eric and Davison, Andrew J.},
    journal={arXiv preprint},
    year={2024},
}      
```
