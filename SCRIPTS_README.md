# MASt3R-SLAM Setup Scripts

Convenience scripts for setting up MASt3R-SLAM on Windows and other platforms.

## Windows Setup Scripts (.bat)

These scripts simplify the setup process on Windows. They must be run from the MASt3R-SLAM root directory.

### setup.bat

**Purpose:** Complete setup of MASt3R-SLAM with all dependencies.

**What it does:**
1. Checks Python is installed
2. Checks NVIDIA GPU is available
3. Prompts you to choose CUDA version (11.8, 12.1, or 12.4)
4. Installs PyTorch with matching CUDA version via conda
5. Installs MASt3R dependencies
6. Installs in3d dependencies
7. Compiles C++ and CUDA extensions
8. Optionally installs torchcodec for faster video loading

**Prerequisites:**
- Conda installed and in PATH
- NVIDIA GPU drivers installed
- CUDA toolkit installed (check with `nvcc --version`)
- 20+ GB free disk space

**Usage:**
```batch
cd C:\path\to\MASt3R-SLAM
setup.bat
```

**Expected output:**
- Prompts for CUDA version
- Installation progress messages
- Final message: "Setup completed successfully!"

**Next steps:**
- Run `get_models.bat` to download model checkpoints
- Run `examples.bat` to see available datasets

---

### get_models.bat

**Purpose:** Download pre-trained model checkpoints from Naver Labs.

**What it does:**
1. Creates `checkpoints/` directory
2. Downloads 3 model files using PowerShell (total ~5 GB):
   - `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth` (2.3 GB)
   - `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth` (1.5 GB)
   - `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl` (200 MB)
3. Verifies all files downloaded successfully
4. Provides manual download links if needed

**Prerequisites:**
- PowerShell 5.1+ (built into Windows 10+)
- Internet connection (100+ Mbps recommended for 1-2 hour download)
- ~6 GB free disk space

**Usage:**
```batch
get_models.bat
```

**Expected output:**
- Download progress from PowerShell
- Verification checklist showing [OK] or [MISSING]
- Checkpoints location: `C:\...\MASt3R-SLAM\checkpoints`

**Troubleshooting:**
- If downloads timeout: Try again later or increase timeout in script
- If downloads fail: Use manual links provided or `wget` from WSL/Git Bash
- To resume incomplete download: Delete partial file and run again

---

### build.bat

**Purpose:** Rebuild C++ and CUDA extensions (used after source changes).

**What it does:**
1. Checks Python is installed
2. Checks NVIDIA GPU availability
3. Displays detected GPU info
4. Compiles C++ and CUDA backend extensions
5. Reports success or errors

**Prerequisites:**
- PyTorch already installed
- CUDA toolkit in PATH
- Python environment activated

**Usage:**
```batch
build.bat
```

**When to run:**
- After cloning the repository
- After modifying `mast3r_slam/backend/src/` files
- If you get import errors for `mast3r_slam_backends`

**Expected output:**
- Compilation progress messages
- Final message: "Build completed successfully!"

---

### examples.bat

**Purpose:** Show available example datasets and how to use them.

**What it does:**
1. Checks models are downloaded
2. Displays 6 example usage scenarios:
   - TUM-RGBD dataset (RGB-D indoor sequences)
   - 7-Scenes dataset (indoor with ground truth)
   - EuRoC dataset (drone sequences with IMU)
   - ETH3D dataset (high-quality SLAM benchmark)
   - Live RealSense camera
   - Custom video or image folder
3. Shows exact commands to run
4. Explains dataset sizes and download times

**Prerequisites:**
- Models already downloaded (run `get_models.bat` first)

**Usage:**
```batch
examples.bat
```

**Output examples:**
- Command: `python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_room/ --config config/calib.yaml`
- Command: `python main.py --dataset C:/path/to/video.mp4 --config config/base.yaml`

---

## Cross-Platform Dataset Download

### download_datasets.py

**Purpose:** Download benchmark datasets (works on Windows, Linux, macOS).

**Supports:**
- TUM-RGBD Dataset (~1-2 GB, 9 sequences)
- EuRoC Dataset (~15 GB, 11 sequences)
- 7-Scenes Dataset (info only, manual download required)
- ETH3D Dataset (info only, requires registration)

**What it does:**
1. Creates `datasets/` directory structure
2. Downloads tar.gz or zip files
3. Automatically extracts archives
4. Skips already-downloaded files (resume support)
5. Shows progress and final location

**Prerequisites:**
- Python 3.6+
- Internet connection
- 20+ GB free disk space (for all datasets)

**Installation:**
No special installation needed - uses Python standard library.

**Usage:**

```bash
# Download TUM-RGBD dataset (~1-2 GB, 10 minutes)
python download_datasets.py --tum

# Download EuRoC dataset (~15 GB, 1 hour)
python download_datasets.py --euroc

# Show information about all datasets
python download_datasets.py --list

# Download both TUM and EuRoC
python download_datasets.py --tum --euroc

# Show help
python download_datasets.py --help
```

**Output example:**
```
=== TUM-RGBD Dataset ===
Downloading: https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz
Downloaded to: datasets/tum/rgbd_dataset_freiburg1_room.tgz
Extracting: datasets/tum/rgbd_dataset_freiburg1_room.tgz
Extracted to: datasets/tum/

Datasets saved to: datasets/
```

**Resume interrupted downloads:**
The script checks if files already exist and skips them. Safe to re-run.

---

## Quick Start Examples

### Scenario 1: First-time setup on Windows

```batch
REM 1. Clone repo
git clone https://github.com/rmurai0610/MASt3R-SLAM.git --recursive
cd MASt3R-SLAM

REM 2. Activate conda environment
conda create -n mast3r-slam python=3.11
conda activate mast3r-slam

REM 3. Run setup (interactive, prompts for CUDA version)
setup.bat

REM 4. Download models (takes 1-2 hours)
get_models.bat

REM 5. Download example dataset (takes 10-30 minutes)
python download_datasets.py --tum

REM 6. Run first example
python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_room/ --config config/calib.yaml
```

### Scenario 2: Test with video file

```bash
REM Already have models and setup complete
REM Run on custom MP4 video
python main.py --dataset C:/path/to/myvideo.mp4 --config config/base.yaml
```

### Scenario 3: Rebuild after source changes

```batch
REM After modifying C++ code in mast3r_slam/backend/src/
build.bat

REM Then re-run
python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_room/ --config config/calib.yaml
```

### Scenario 4: Download and evaluate multiple datasets

```bash
python download_datasets.py --tum
python download_datasets.py --euroc
bash scripts/eval_tum.sh
bash scripts/eval_euroc.sh
```

---

## Environment Setup Notes

### CUDA Version Selection

The setup script requires you to choose a CUDA version. To find your system's CUDA version:

```bash
# Check installed CUDA version
nvcc --version
```

Choose the setup option matching your CUDA version:
- CUDA 11.8
- CUDA 12.1  
- CUDA 12.4
- CUDA 12.8

If your CUDA version doesn't match exactly, choose the closest one (e.g., 12.0 → choose 12.1 or 12.4).

### Conda Environment

All scripts expect the conda environment to be activated:

```bash
conda activate mast3r-slam
```

Some scripts can automatically detect conda on PATH. If issues occur, manually activate.

### GPU Memory Requirements

- Minimum: 8 GB VRAM (will be slow)
- Recommended: 24 GB VRAM (RTX 3090, RTX 4090)
- Tested on: RTX 4090 (24 GB)

---

## Troubleshooting

### "Cannot find CUDA in PATH"

**Solution:**
1. Install CUDA Toolkit from NVIDIA
2. Run `nvcc --version` to verify installation
3. If `nvcc` not found, add CUDA to PATH:
   - Find CUDA installation folder (usually `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X`)
   - Add `bin` folder to Windows PATH
   - Restart terminal

### "pytorch-cuda installation failed"

**Possible causes:**
- Wrong CUDA version in setup.bat
- Conda not in PATH
- Network timeout during download

**Solutions:**
1. Verify CUDA version: `nvcc --version`
2. Run setup.bat again with correct version
3. Ensure conda is available: `conda --version`
4. Check internet connection

### "mast3r_slam_backends module not found"

**Cause:** C++ extensions not compiled

**Solution:** Run `build.bat`

### Download timeouts

**Cause:** Large files timing out during download

**Solutions:**
1. Check internet connection (>10 Mbps)
2. Run script again (resume interrupted downloads)
3. Try later if server is busy
4. Manually download to `datasets/` folder

---

## File Locations

After successful setup:

```
MASt3R-SLAM/
├── checkpoints/                    # Model weights (5 GB)
│   ├── MASt3R_ViTLarge_*.pth
│   └── MASt3R_ViTLarge_*.pkl
├── datasets/                       # Benchmark datasets (25+ GB)
│   ├── tum/                        # TUM-RGBD sequences
│   └── euroc/                      # EuRoC sequences
├── mast3r_slam/
│   ├── backend/
│   │   └── src/                    # C++ and CUDA source
│   └── ...
├── setup.bat                       # Setup script
├── get_models.bat                  # Download models
├── build.bat                       # Rebuild extensions
├── examples.bat                    # Show examples
├── download_datasets.py            # Python dataset downloader
└── main.py                         # Main entry point
```

---

## Next Steps

1. Complete setup with `setup.bat`
2. Download models with `get_models.bat`
3. Download sample data with `python download_datasets.py --tum`
4. Run example: `python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_room/ --config config/calib.yaml`
5. See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed documentation
6. See [README.md](README.md) for project overview

---

## Support

For issues:
1. Check troubleshooting section above
2. See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed setup instructions
3. Check GitHub issues: https://github.com/rmurai0610/MASt3R-SLAM/issues
4. See project page: https://edexheim.github.io/mast3r-slam/
