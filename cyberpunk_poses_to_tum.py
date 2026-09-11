#!/usr/bin/env python3
"""Convert cyberpunk_results/poses.json (another model's own estimated c2w
poses, NOT real ground truth) into a TUM-format trajectory.txt for evo
comparison against MASt3R-SLAM output. extrinsic is already standard
column-vector c2w (3x4, rotation | translation), 30fps frame index as time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("poses_json", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    out = args.out or args.poses_json.parent / "poses_trajectory.txt"

    d = json.load(open(args.poses_json))
    lines = []
    for f in d["frames"]:
        t = f["frame"] / args.fps
        c2w = np.array(f["extrinsic"], dtype=np.float64)  # 3x4, standard c2w
        R = c2w[:3, :3]
        tx, ty, tz = c2w[:3, 3]
        qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()
        lines.append(f"{t:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}: {len(lines)} poses")


if __name__ == "__main__":
    main()
