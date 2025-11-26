#!/bin/bash

# Script to run MASt3R-SLAM in Docker container

# Default dataset if no argument provided
DATASET=${1:-"data/yourvideo.mp4"}

# Navigate to the parent directory (MASt3R-SLAM root)
cd "$(dirname "$0")/.."

# First check and download checkpoints if they don't exist
./scripts/download_checkpoints.sh

# Extract filename without path and extension for output naming
FILENAME=$(basename "$DATASET")
FILENAME_NO_EXT="${FILENAME%.*}"

echo "Starting MASt3R-SLAM processing..."
echo "Dataset: $DATASET"
echo "This may take several minutes depending on the video length..."
echo "Results will be saved to ./output/ directory:"
echo "  - ${FILENAME_NO_EXT}.txt (camera trajectory)"
echo "  - ${FILENAME_NO_EXT}.ply (3D point cloud)"
echo "  - keyframes/ (keyframe images)"

# Run the docker container and execute the command with saving enabled
# Using -T flag to disable pseudo-TTY allocation for cleaner output
# Save results to output directory which is mapped to host ./output
docker compose exec -T mast3r-slam bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate mast3r-slam && python main.py --dataset $DATASET --config config/base.yaml --no-viz --save-as output"