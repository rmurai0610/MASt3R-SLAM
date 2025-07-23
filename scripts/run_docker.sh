#!/bin/bash

# Script to run MASt3R-SLAM in Docker container

# Navigate to the parent directory (MASt3R-SLAM root)
cd "$(dirname "$0")/.."

# First check and download checkpoints if they don't exist
./scripts/download_checkpoints.sh

echo "Starting MASt3R-SLAM processing..."
echo "This may take several minutes depending on the video length..."
echo "Results will be saved to ./output/ directory:"
echo "  - otowa_koregaseikai.txt (camera trajectory)"
echo "  - otowa_koregaseikai.ply (3D point cloud)"
echo "  - keyframes/ (keyframe images)"

# Run the docker container and execute the command with saving enabled
# Using -T flag to disable pseudo-TTY allocation for cleaner output
# Save results to output directory which is mapped to host ./output
docker compose exec -T mast3r-slam bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate mast3r-slam && python main.py --dataset data/otowa_koregaseikai.mov --config config/base.yaml --no-viz --save-as output"