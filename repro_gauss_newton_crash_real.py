#!/usr/bin/env python3
"""Replay the exact real tensors that triggered the gauss_newton_rays crash
(dumped via DUMP_GN_INPUTS=1) to get a fast, faithful, minimal repro.
"""
import torch
import mast3r_slam_backends

d = torch.load("gn_inputs_dump.pt")
print("Loaded real crash-triggering inputs:")
for k, v in d.items():
    if torch.is_tensor(v):
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
    else:
        print(f"  {k}: {v}")

mast3r_slam_backends.gauss_newton_rays(
    d["pose_data"].to(d["Xs"].device),
    d["Xs"],
    d["Cs"],
    d["ii"],
    d["jj"],
    d["idx_ii2jj"],
    d["valid_match"],
    d["Q_ii2jj"],
    d["sigma_ray"],
    d["sigma_dist"],
    d["C_thresh"],
    d["Q_thresh"],
    d["max_iter"],
    d["delta_thresh"],
)
torch.cuda.synchronize()
print("SUCCESS: no crash.")
