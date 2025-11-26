#!/bin/bash

# Script to run MASt3R-SLAM in Docker container with low memory settings

# Default dataset if no argument provided
DATASET=${1:-"data/yourvideo.mp4"}

# Navigate to the parent directory (MASt3R-SLAM root)
cd "$(dirname "$0")/.."

# First check and download checkpoints if they don't exist
./scripts/download_checkpoints.sh

# Extract filename without path and extension for output naming
FILENAME=$(basename "$DATASET")
FILENAME_NO_EXT="${FILENAME%.*}"

echo "Starting MASt3R-SLAM processing with low memory settings..."
echo "Dataset: $DATASET"
echo "Using:"
echo "  - Frame subsampling: 3 (every 3rd frame)"
echo "  - Image downsampling: 2 (960x540 resolution)"
echo "  - Single threading to reduce memory usage"
echo "This may take several minutes depending on the video length..."
echo "Results will be saved to ./output/ directory:"
echo "  - ${FILENAME_NO_EXT}.txt (camera trajectory)"
echo "  - ${FILENAME_NO_EXT}.ply (3D point cloud)"
echo "  - keyframes/ (keyframe images)"

# Set PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Run the docker container and execute the command with --no-viz flag, low memory config, and saving enabled
docker compose exec -T mast3r-slam bash -c "export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' && source /opt/conda/etc/profile.d/conda.sh && conda activate mast3r-slam && python main.py --dataset $DATASET --config config/low_memory.yaml --no-viz --save-as output"