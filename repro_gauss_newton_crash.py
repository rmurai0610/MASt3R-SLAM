#!/usr/bin/env python3
"""Minimal standalone repro of the gauss_newton_rays CUDA crash -- calls the
compiled kernel directly with synthetic data matching the exact shapes/
dtypes/devices seen in the real crash (num_poses=2, num_points=147456,
num_edges=2), skipping all model loading / video decoding / tracking.

Run under compute-sanitizer for a fast, targeted repro:
    "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\bin\\compute-sanitizer.bat" --launch-timeout 0 .venv\\Scripts\\python.exe repro_gauss_newton_crash.py
"""
import torch
import mast3r_slam_backends

device = "cuda"
num_poses = 2
num_points = 147456
num_edges = 2

# pose_data: [tx,ty,tz, qx,qy,qz,qw, scale] per pose (lietorch Sim3 embedded layout)
pose_data = torch.zeros(num_poses, 8, device=device, dtype=torch.float32)
pose_data[:, 6] = 1.0  # qw = 1 (identity rotation)
pose_data[:, 7] = 1.0  # scale = 1
pose_data[1, 0] = 0.1  # give pose 1 a small translation so it's not identical to pose 0

Xs = torch.randn(num_poses, num_points, 3, device=device, dtype=torch.float32)
Xs[:, :, 2] += 3.0  # push points in front of the camera (positive z-ish)

Cs = torch.rand(num_poses, num_points, 1, device=device, dtype=torch.float32) * 5.0

ii = torch.tensor([0, 1], device=device, dtype=torch.int64)
jj = torch.tensor([1, 0], device=device, dtype=torch.int64)

idx_ii2jj = torch.randint(0, num_points, (num_edges, num_points), device=device, dtype=torch.int64)
valid_match = torch.ones(num_edges, num_points, 1, device=device, dtype=torch.bool)
Q_ii2jj = torch.rand(num_edges, num_points, 1, device=device, dtype=torch.float32) * 5.0

sigma_ray = 0.003
sigma_dist = 10.0
C_thresh = 0.0
Q_thresh = 1.5
max_iter = 10
delta_thresh = 1e-8

print("Calling gauss_newton_rays with synthetic data matching the real crash's shapes...")
print(f"  pose_data={tuple(pose_data.shape)} Xs={tuple(Xs.shape)} Cs={tuple(Cs.shape)}")
print(f"  ii={tuple(ii.shape)} jj={tuple(jj.shape)} idx_ii2jj={tuple(idx_ii2jj.shape)}")
print(f"  valid_match={tuple(valid_match.shape)} Q_ii2jj={tuple(Q_ii2jj.shape)}")

mast3r_slam_backends.gauss_newton_rays(
    pose_data,
    Xs,
    Cs,
    ii,
    jj,
    idx_ii2jj,
    valid_match,
    Q_ii2jj,
    sigma_ray,
    sigma_dist,
    C_thresh,
    Q_thresh,
    max_iter,
    delta_thresh,
)
torch.cuda.synchronize()
print("SUCCESS: no crash. pose_data after solve:")
print(pose_data)
