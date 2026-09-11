#!/usr/bin/env python3
"""View a MASt3R-SLAM reconstruction: point cloud (.ply) + camera trajectory
(.txt, TUM format) as small camera frustums, in an interactive open3d window.
Optionally overlay a ground-truth trajectory, Sim(3)-aligned + scale-corrected
onto the estimate (same alignment evo_ape/evo_traj use) so it's meaningfully
comparable, not just plotted in its own raw coordinate frame.

Usage:
    python view_result.py logs/<run>/<seq>.ply logs/<run>/<seq>.txt
    python view_result.py logs/<run>/<seq>.ply logs/<run>/<seq>.txt --groundtruth path/to/groundtruth.txt
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


def trajectory_line(traj_pts: list[np.ndarray], color: list[float]) -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(traj_pts))
    ls.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(traj_pts) - 1)])
    ls.colors = o3d.utility.Vector3dVector([color] * (len(traj_pts) - 1))
    return ls


def add_trajectory_geoms(
    geoms: list, poses: list[np.ndarray], stride: int, scale: float,
    start_color: list[float], mid_color: list[float], end_color: list[float],
) -> None:
    traj_pts = []
    for i, T in enumerate(poses):
        traj_pts.append(T[:3, 3])
        if i % stride == 0:
            color = start_color if i == 0 else end_color if i == len(poses) - 1 else mid_color
            geoms.append(make_camera_frustum(T, scale, color))
    if len(traj_pts) > 1:
        geoms.append(trajectory_line(traj_pts, mid_color))


def align_gt_to_est(gt_path: Path, est_path: Path) -> list[np.ndarray]:
    """Sim(3)-align + scale-correct ground truth onto the estimate (matches
    evo_ape/evo_traj -a -s), returning the ALIGNED ground truth poses so it
    overlays meaningfully on the estimate's coordinate frame."""
    from evo.tools import file_interface
    from evo.core import sync

    traj_est = file_interface.read_tum_trajectory_file(str(est_path))
    traj_gt = file_interface.read_tum_trajectory_file(str(gt_path))
    traj_gt_sync, traj_est_sync = sync.associate_trajectories(traj_gt, traj_est)
    # align the ORIGINAL full-length gt onto est using the sync'd correspondence
    r, t, s = traj_gt_sync.align(traj_est_sync, correct_scale=True)
    traj_gt.scale(s)
    traj_gt.transform(np.block([[r, t.reshape(3, 1)], [0, 0, 0, 1]]))

    poses = []
    for T in traj_gt.poses_se3:
        poses.append(np.array(T))
    return poses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ply_path", type=Path)
    ap.add_argument("traj_path", type=Path)
    ap.add_argument("--groundtruth", type=Path, default=None)
    ap.add_argument("--frustum-scale", type=float, default=0.05)
    ap.add_argument("--frustum-stride", type=int, default=1, help="show every Nth camera")
    args = ap.parse_args()

    pcd = o3d.io.read_point_cloud(str(args.ply_path))
    print(f"Loaded point cloud: {len(pcd.points)} points")

    poses = load_tum_trajectory(args.traj_path)
    print(f"Loaded estimated trajectory: {len(poses)} poses")

    geoms = [pcd]
    # estimate: green=start, yellow=mid, red=end
    add_trajectory_geoms(
        geoms, poses, args.frustum_stride, args.frustum_scale,
        start_color=[0.0, 1.0, 0.0], mid_color=[1.0, 1.0, 0.0], end_color=[1.0, 0.0, 0.0],
    )

    title = f"{args.ply_path.name}: est={len(poses)} poses (green=start,red=end)"
    if args.groundtruth is not None:
        gt_poses = align_gt_to_est(args.groundtruth, args.traj_path)
        print(f"Loaded + Sim(3)-aligned ground truth: {len(gt_poses)} poses")
        # ground truth: all white/gray, distinct from the colored estimate
        add_trajectory_geoms(
            geoms, gt_poses, max(1, len(gt_poses) // max(1, len(poses) // args.frustum_stride)),
            args.frustum_scale * 0.7,
            start_color=[0.6, 0.6, 0.6], mid_color=[0.6, 0.6, 0.6], end_color=[0.6, 0.6, 0.6],
        )
        title += f" vs groundtruth={len(gt_poses)} poses (gray, Sim(3)-aligned)"

    o3d.visualization.draw_geometries(geoms, window_name=title)


if __name__ == "__main__":
    main()
