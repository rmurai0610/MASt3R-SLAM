#!/bin/bash

# Script to clean GPU memory and run MASt3R-SLAM with low memory settings

# Navigate to the parent directory (MASt3R-SLAM root)
cd "$(dirname "$0")/.."

echo "Cleaning GPU memory..."

# Kill any existing Python processes that might be using GPU memory
pkill -f "python.*main.py" || true
pkill -f "conda.*python" || true

# Wait for processes to terminate
sleep 3

# Check current GPU memory usage
echo "Current GPU memory usage:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits

# First check and download checkpoints if they don't exist
./scripts/download_checkpoints.sh

echo "Starting MASt3R-SLAM processing with aggressive memory optimization..."
echo "Using:"
echo "  - Frame subsampling: 5 (every 5th frame)"
echo "  - Image downsampling: 3 (640x360 resolution)"
echo "  - Single threading to reduce memory usage"
echo "  - Reduced window size and retrieval settings"
echo "This may take several minutes depending on the video length..."

# Set PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Run the docker container and execute the command with --no-viz flag and ultra low memory config
docker compose exec -T mast3r-slam bash -c "export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' && source /opt/conda/etc/profile.d/conda.sh && conda activate mast3r-slam && python main.py --dataset data/otowa_koregaseikai.mov --config config/ultra_low_memory.yaml --no-viz"