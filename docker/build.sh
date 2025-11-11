#!/bin/bash
set -e

DOCKER_BUILDKIT=0 docker compose build --progress plain
