#!/bin/bash

# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints/

# Check and download MASt3R checkpoints only if they don't exist
echo "Checking MASt3R checkpoints..."

if [ ! -f "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" ]; then
    echo "Downloading MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth..."
    wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
else
    echo "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth already exists, skipping download."
fi

if [ ! -f "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth" ]; then
    echo "Downloading MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth..."
    wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
else
    echo "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth already exists, skipping download."
fi

if [ ! -f "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl" ]; then
    echo "Downloading MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl..."
    wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/
else
    echo "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl already exists, skipping download."
fi

echo "Checkpoint check complete!"