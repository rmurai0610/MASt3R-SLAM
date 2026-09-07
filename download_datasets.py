#!/usr/bin/env python3
"""Download MASt3R-SLAM datasets for Windows and Unix."""

import argparse
import os
import sys
import zipfile
import tarfile
import urllib.request
from pathlib import Path
from urllib.error import URLError


def create_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def download_file(url, dest_path, chunk_size=8192):
    """Download a file from URL with progress indication."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Downloading: {url}")
        urllib.request.urlretrieve(url, dest_path)
        print(f"Downloaded to: {dest_path}")
        return True
    except URLError as e:
        print(f"Error downloading {url}: {e}")
        return False


def extract_tar(file_path, extract_to):
    """Extract tar.gz file."""
    print(f"Extracting: {file_path}")
    try:
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(path=extract_to)
        print(f"Extracted to: {extract_to}")
        return True
    except Exception as e:
        print(f"Error extracting {file_path}: {e}")
        return False


def extract_zip(file_path, extract_to):
    """Extract zip file."""
    print(f"Extracting: {file_path}")
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(path=extract_to)
        print(f"Extracted to: {extract_to}")
        return True
    except Exception as e:
        print(f"Error extracting {file_path}: {e}")
        return False


def download_tum():
    """Download TUM-RGBD Dataset."""
    print("\n=== TUM-RGBD Dataset ===")
    dest = create_dir("datasets/tum")

    urls = [
        "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_360.tgz",
        "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_floor.tgz",
        "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz",
        "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk2.tgz",
        "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz",
        "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_plant.tgz",
        "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_teddy.tgz",
        "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz",
        "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_rpy.tgz",
    ]

    for url in urls:
        file_name = url.split('/')[-1]
        file_path = os.path.join(dest, file_name)

        if os.path.exists(file_path):
            print(f"Already downloaded: {file_path}")
        else:
            if not download_file(url, file_path):
                continue

        if file_path.endswith('.tgz'):
            extract_tar(file_path, dest)


def download_euroc():
    """Download EuRoC Dataset."""
    print("\n=== EuRoC Dataset ===")
    dest = create_dir("datasets/euroc")

    urls = [
        "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_01_easy/V1_01_easy.zip",
        "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_02_medium/V1_02_medium.zip",
        "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_03_difficult/V1_03_difficult.zip",
        "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_01_easy/V2_01_easy.zip",
        "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_02_medium/V2_02_medium.zip",
        "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_03_difficult/V2_03_difficult.zip",
        "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip",
        "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_02_easy/MH_02_easy.zip",
        "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_03_medium/MH_03_medium.zip",
        "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_04_difficult/MH_04_difficult.zip",
        "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_05_difficult/MH_05_difficult.zip",
    ]

    for url in urls:
        file_name = url.split('/')[-1]
        file_path = os.path.join(dest, file_name)

        if os.path.exists(file_path):
            print(f"Already downloaded: {file_path}")
        else:
            if not download_file(url, file_path):
                continue

        if file_path.endswith('.zip'):
            extract_path = os.path.join(dest, file_name[:-4])
            extract_zip(file_path, extract_path)


def download_7scenes():
    """Download 7-Scenes Dataset."""
    print("\n=== 7-Scenes Dataset ===")
    print("Note: 7-Scenes requires manual download from: https://www.microsoft.com/en-us/download/details.aspx?id=52283")
    print("Extract the downloaded files to: datasets/7_scenes/")


def download_eth3d():
    """Download ETH3D SLAM Dataset."""
    print("\n=== ETH3D SLAM Dataset ===")
    print("Note: ETH3D requires registration at: https://www.eth3d.net/slam_datasets")
    print("Download datasets and extract to: datasets/eth3d/")


def list_datasets():
    """List available datasets with sizes."""
    print("\nAvailable Datasets:")
    print("==================")
    print("\n1. TUM-RGBD (Small, ~1-2 GB)")
    print("   - Indoor RGB-D sequences")
    print("   - ~10 scenes with ground truth")
    print("   - Good for testing")

    print("\n2. 7-Scenes (Medium, ~10 GB)")
    print("   - Indoor scenes: chess, fire, heads, office, pumpkin, redkitchen, stairs")
    print("   - High-quality ground truth")
    print("   - Requires manual download")

    print("\n3. EuRoC (Medium, ~15 GB)")
    print("   - Drone flight sequences with IMU data")
    print("   - 11 sequences with varying difficulty")
    print("   - High-quality camera calibration")

    print("\n4. ETH3D (Large, ~50 GB)")
    print("   - High-quality SLAM benchmark")
    print("   - Requires registration")


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for MASt3R-SLAM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_datasets.py --tum          # Download TUM-RGBD
  python download_datasets.py --euroc        # Download EuRoC
  python download_datasets.py --list         # List all datasets
  python download_datasets.py --all          # Download all available
        """
    )

    parser.add_argument('--tum', action='store_true', help='Download TUM-RGBD dataset')
    parser.add_argument('--euroc', action='store_true', help='Download EuRoC dataset')
    parser.add_argument('--7scenes', dest='seven_scenes', action='store_true', help='Info about 7-Scenes dataset')
    parser.add_argument('--eth3d', action='store_true', help='Info about ETH3D dataset')
    parser.add_argument('--list', action='store_true', help='List all datasets')
    parser.add_argument('--all', action='store_true', help='Download all available datasets')

    args = parser.parse_args()

    if not any([args.tum, args.euroc, args.seven_scenes, args.eth3d, args.list, args.all]):
        parser.print_help()
        return

    if args.list:
        list_datasets()
        return

    if args.tum or args.all:
        download_tum()

    if args.euroc or args.all:
        download_euroc()

    if args.seven_scenes:
        download_7scenes()

    if args.eth3d:
        download_eth3d()

    print("\nDatasets saved to: datasets/")
    print("\nRun examples with:")
    print("  python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_room/ --config config/calib.yaml")
    print("  python main.py --dataset datasets/euroc/V1_01_easy/ --config config/base.yaml")


if __name__ == "__main__":
    main()
