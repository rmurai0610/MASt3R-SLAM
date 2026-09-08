#!/usr/bin/env python3
"""Convert an isekai *_data.npy camera replay recording into a TUM-format
groundtruth.txt (timestamp tx ty tz qx qy qz qw) for evo comparison against
MASt3R-SLAM output.

isekai's camera_to_world is stored in Unreal Engine's row-vector convention
(translation in the LAST ROW, not last column) and centimeters -- this
transposes to standard column-vector convention and converts cm -> m.

Usage:
    python isekai_to_tum_gt.py C:\\workspace\\data\\isekai\\fdse\\<...>_data.npy [--out groundtruth.txt]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_npy", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default: <data_npy dir>/groundtruth.txt")
    args = ap.parse_args()

    out = args.out or args.data_npy.parent / "groundtruth.txt"

    data = np.load(args.data_npy, allow_pickle=True)
    lines = []
    for row in data:
        t = float(row["pts"])
        # Unreal row-vector c2w -> standard column-vector c2w: transpose.
        c2w = row["camera_to_world"].astype(np.float64).T
        R = c2w[:3, :3]
        # cm -> m
        tx, ty, tz = c2w[:3, 3] / 100.0
        qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()
        lines.append(f"{t:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}: {len(lines)} poses, t=[{data[0]['pts']:.2f}, {data[-1]['pts']:.2f}]s")


if __name__ == "__main__":
    main()
