# MASt3R-SLAM Setup Guide

This guide covers setting up MASt3R-SLAM on Windows, Linux, and WSL.

## Prerequisites

- Python 3.11+ (from https://www.python.org/downloads/)
- NVIDIA GPU with CUDA support (tested on RTX 4090)
- NVIDIA CUDA Toolkit (11.8, 12.1, 12.4, or 12.8) from https://developer.nvidia.com/cuda-downloads
- Git with submodules support
- ~30 GB free disk space (for models and datasets)

## Quick Start (Windows)

### Step 1: Clone the Repository

```bash
git clone https://github.com/rmurai0610/MASt3R-SLAM.git --recursive
cd MASt3R-SLAM
```

### Step 2: Run Setup Script

```batch
setup.bat
```

This script will automatically:
1. Create a Python virtual environment in `.venv/`
2. Activate the environment
3. Check your CUDA Toolkit installation
4. Prompt you to choose CUDA version (11.8, 12.1, 12.4, or 12.8)
5. Install PyTorch with matching CUDA version
6. Install MASt3R dependencies from `thirdparty/mast3r`
7. Install in3d dependencies from `thirdparty/in3d`
8. Compile MASt3R-SLAM C++ and CUDA extensions
9. Verify everything installed correctly
10. Optionally install torchcodec for faster video loading

**⚠️ Important:** Before running setup.bat:
1. Install Python 3.11+ from https://www.python.org/downloads/
2. Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
3. Verify CUDA is installed: Run `nvcc --version` in Command Prompt
4. Know which CUDA version you have (11.8, 12.1, 12.4, or 12.8)

### Step 4: Download Models

```batch
get_models.bat
```

This downloads ~5 GB of pre-trained checkpoints to the `checkpoints/` directory:
- `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth` (2.3 GB)
- `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth` (1.5 GB)
- `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl` (200 MB)

⚠️ **Note:** Downloads can be slow. If they fail, manually download from:
```
https://download.europe.naverlabs.com/ComputerVision/MASt3R/
```

### Step 5: Download Example Datasets

```batch
examples.bat
```

This shows available datasets and how to download them. To download specific datasets:

```bash
# Download TUM-RGBD (smallest, ~1-2 GB, ~10 min)
python download_datasets.py --tum

# Download EuRoC (medium, ~15 GB, ~1 hour)
python download_datasets.py --euroc

# Show info about other datasets
python download_datasets.py --list
```

### Step 6: Test with Example Data

```bash
# Test with TUM-RGBD Room sequence
python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_room/ --config config/calib.yaml
```

## Available Scripts

### setup.bat
**Windows only.** Interactive setup that installs all dependencies.
- Checks Python and CUDA availability
- Prompts for CUDA version (11.8, 12.1, or 12.4)
- Installs PyTorch, MASt3R, in3d, and builds C++ extensions
- Installs optional torchcodec

### get_models.bat
**Windows only.** Downloads pre-trained model checkpoints.
- Creates `checkpoints/` directory
- Downloads 3 model files (~5 GB total)
- Verifies downloads
- Shows manual download links if automatic download fails

### build.bat
**Windows only.** Rebuilds C++ and CUDA extensions.
- Checks Python and CUDA availability
- Displays detected GPU
- Compiles extensions in-place

### examples.bat
**Windows only.** Shows available example datasets and running instructions.

### download_datasets.py
**Cross-platform.** Python script to download datasets on any OS.

```bash
# Download TUM-RGBD dataset
python download_datasets.py --tum

# Download EuRoC dataset
python download_datasets.py --euroc

# Show dataset info
python download_datasets.py --list

# Show help
python download_datasets.py --help
```

## Linux/WSL Setup

### Step 1: Clone and Environment Setup

```bash
git clone https://github.com/rmurai0610/MASt3R-SLAM.git --recursive
cd MASt3R-SLAM

python3.11 -m venv .venv
source .venv/bin/activate
```

### Step 2: Upgrade Build Tools

```bash
pip install --upgrade pip setuptools wheel
```

### Step 3: Install PyTorch

Check CUDA version:
```bash
nvcc --version
```

Install matching PyTorch (example for CUDA 12.8):
```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

For other CUDA versions:
- CUDA 11.8: `--index-url https://download.pytorch.org/whl/cu118`
- CUDA 12.1: `--index-url https://download.pytorch.org/whl/cu121`
- CUDA 12.4: `--index-url https://download.pytorch.org/whl/cu124`

### Step 4: Install Dependencies

```bash
pip install -e thirdparty/mast3r --no-build-isolation
pip install -e thirdparty/in3d --no-build-isolation
pip install --no-build-isolation -e .
pip install torchcodec==0.1  # optional
```

### Step 5: Download Models

```bash
bash scripts/download_tum.sh  # Downloads TUM dataset
```

Or manually:
```bash
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/
```

### Step 6: Download Datasets

Use the Python script (cross-platform):
```bash
python download_datasets.py --tum    # ~1-2 GB
python download_datasets.py --euroc  # ~15 GB
```

Or the bash scripts:
```bash
bash scripts/download_tum.sh
bash scripts/download_euroc.sh
bash scripts/download_7_scenes.sh
bash scripts/download_eth3d.sh
```

### Step 7: Run Examples

```bash
# TUM-RGBD with calibration
python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_room/ --config config/calib.yaml

# EuRoC without calibration
python main.py --dataset datasets/euroc/V1_01_easy/ --config config/base.yaml

# Video or image folder
python main.py --dataset path/to/video.mp4 --config config/base.yaml
python main.py --dataset path/to/images/ --config config/base.yaml

# RealSense camera (live)
python main.py --dataset realsense --config config/base.yaml
```

## Troubleshooting

### CUDA Not Found

**Error:** `Cannot find CUDA in PATH`

**Solution:**
1. Install CUDA Toolkit from NVIDIA
2. Add CUDA to your PATH:
   - Windows: Add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin` to PATH
   - Linux: Add to `~/.bashrc`: `export PATH=/usr/local/cuda/bin:$PATH`
3. Verify: Run `nvcc --version`

### PyTorch Build Error

**Error:** `Fatal error C1083` or similar compilation error

**Possible causes:**
1. Wrong PyTorch CUDA version (doesn't match your CUDA toolkit)
2. Missing Visual Studio Build Tools (Windows)
3. Out of disk space during build

**Solutions:**
- Verify PyTorch CUDA version matches system CUDA: `python -c "import torch; print(torch.version.cuda)"`
- Install Visual Studio Build Tools with C++ support
- Ensure 20+ GB free disk space
- Run `setup.bat` again with correct CUDA version

### Download Timeouts

**Error:** Connection timeout when downloading datasets

**Solutions:**
1. Check internet connection
2. Try again later (server might be busy)
3. For large files, resume with:
   ```bash
   python download_datasets.py --tum  # automatically skips existing files
   ```
4. Manually download from the project website

### RealSense Camera Not Detected

**Requirements:**
- RealSense camera connected via USB 3.0+
- RealSense SDK installed
- PyRealSense2 package installed

**Installation:**
```bash
pip install pyrealsense2
```

### Out of Memory

**Error:** CUDA out of memory during inference

**Solutions:**
1. Close other GPU applications
2. Process smaller videos/images
3. Reduce resolution in config file
4. Use smaller batch sizes in config

## Running Evaluations

Evaluate on benchmark datasets:

```bash
# TUM-RGBD with calibration
bash scripts/eval_tum.sh

# TUM-RGBD without calibration
bash scripts/eval_tum.sh --no-calib

# 7-Scenes
bash scripts/eval_7_scenes.sh
bash scripts/eval_7_scenes.sh --no-calib

# EuRoC
bash scripts/eval_euroc.sh
bash scripts/eval_euroc.sh --no-calib

# ETH3D
bash scripts/eval_eth3d.sh
```

Note: These are bash scripts. On Windows, use WSL or Git Bash.

## Configuration Files

Available configurations in `config/`:

- **base.yaml** - Basic configuration (no calibration)
- **calib.yaml** - With camera calibration
- **eval_calib.yaml** - Evaluation with calibration
- **eval_no_calib.yaml** - Evaluation without calibration
- **eth3d.yaml** - ETH3D dataset specific
- **intrinsics.yaml** - Template for custom camera intrinsics

To use custom camera calibration:
```bash
python main.py --dataset path/to/data --config config/base.yaml --calib config/intrinsics.yaml
```

Edit `config/intrinsics.yaml` with your camera parameters:
```yaml
camera:
  fx: 500.0  # focal length x
  fy: 500.0  # focal length y
  cx: 320.0  # principal point x
  cy: 240.0  # principal point y
  width: 640
  height: 480
```

## Performance Notes

- Tested on RTX 4090
- Real-time performance requires ~60 FPS video input
- Slower GPUs may not achieve real-time speeds
- Processing resolution and frame skip can be tuned in configs

## Next Steps

- Read the [main README.md](README.md) for project details
- Check [project page](https://edexheim.github.io/mast3r-slam/) for more info
- See paper at [arXiv](https://arxiv.org/abs/2412.12392)
- Watch [project video](https://youtu.be/wozt71NBFTQ)

## License and Attribution

Check [CHECKPOINTS_NOTICE](https://github.com/naver/mast3r/blob/mast3r_sfm/CHECKPOINTS_NOTICE) for model licenses and citations.
