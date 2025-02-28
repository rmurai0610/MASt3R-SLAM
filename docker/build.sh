#!/bin/bash
set -e

IMAGE_NAME="mast3r-slam"
TAG="latest"

# DOCKER_BUILDKIT=0 docker build --target deploy -f Dockerfile -t ${IMAGE_NAME}:${TAG} ../
DOCKER_BUILDKIT=0 docker compose build

echo "Docker image ${IMAGE_NAME}:${TAG} built successfully."
