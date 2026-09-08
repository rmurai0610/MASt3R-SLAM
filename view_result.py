#!/usr/bin/env python3
"""View a MASt3R-SLAM reconstruction: point cloud (.ply) + camera trajectory
(.txt, TUM format) as small camera frustums, in an interactive open3d window.

Usage:
    python view_result.py logs/<run>/<seq>.ply logs/<run>/<seq>.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


def load_tum_trajectory(path: Path) -> list[np.ndarray]:
    poses = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        vals = [float(x) for x in line.split()]
        _, tx, ty, tz, qx, qy, qz, qw = vals
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        T[:3, 3] = [tx, ty, tz]
        poses.append(T)
    return poses


def make_camera_frustum(T: np.ndarray, scale: float, color: list[float]) -> o3d.geometry.LineSet:
    # simple pyramid frustum in camera space, +z forward
    pts_cam = np.array(
        [
            [0, 0, 0],
            [0.5, 0.35, 1],
            [-0.5, 0.35, 1],
            [-0.5, -0.35, 1],
            [0.5, -0.35, 1],
        ]
    ) * scale
    pts_world = (T[:3, :3] @ pts_cam.T).T + T[:3, 3]
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts_world)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ply_path", type=Path)
    ap.add_argument("traj_path", type=Path)
    ap.add_argument("--frustum-scale", type=float, default=0.05)
    ap.add_argument("--frustum-stride", type=int, default=1, help="show every Nth camera")
    args = ap.parse_args()

    pcd = o3d.io.read_point_cloud(str(args.ply_path))
    print(f"Loaded point cloud: {len(pcd.points)} points")

    poses = load_tum_trajectory(args.traj_path)
    print(f"Loaded trajectory: {len(poses)} poses")

    geoms = [pcd]
    traj_pts = []
    for i, T in enumerate(poses):
        traj_pts.append(T[:3, 3])
        if i % args.frustum_stride == 0:
            color = [0.0, 1.0, 0.0] if i == 0 else [1.0, 0.0, 0.0] if i == len(poses) - 1 else [1.0, 1.0, 0.0]
            geoms.append(make_camera_frustum(T, args.frustum_scale, color))

    if len(traj_pts) > 1:
        traj_line = o3d.geometry.LineSet()
        traj_line.points = o3d.utility.Vector3dVector(np.array(traj_pts))
        traj_line.lines = o3d.utility.Vector2iVector(
            [[i, i + 1] for i in range(len(traj_pts) - 1)]
        )
        traj_line.colors = o3d.utility.Vector3dVector([[0.2, 0.6, 1.0]] * (len(traj_pts) - 1))
        geoms.append(traj_line)

    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"{args.ply_path.name} + {len(poses)} cameras (green=start, red=end)",
    )


if __name__ == "__main__":
    main()
