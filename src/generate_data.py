"""Generate the synthetic dataset: random poses -> 42 projected features -> pose label.

Usage:
    python src/generate_data.py --n 400000
Output:
    data/dataset.npz with X (N, 42), y (N,), poses (N, 6), train_idx, val_idx
"""
import argparse
import os
import time

import numpy as np

import config as C
from simulator import sample_poses, project, pose_to_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400_000, help="number of samples")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "dataset.npz"))
    args = ap.parse_args()

    rng = np.random.default_rng(C.SEED)
    t0 = time.time()

    poses = sample_poses(args.n, rng)
    X = np.empty((args.n, 42), dtype=np.float32)
    for s in range(0, args.n, 50_000):                   # chunked to keep memory low
        X[s:s + 50_000] = project(poses[s:s + 50_000], noise=True, rng=rng)
    y = pose_to_label(poses)

    perm = rng.permutation(args.n)
    n_val = int(args.n * args.val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, poses=poses.astype(np.float32),
                        train_idx=train_idx, val_idx=val_idx)

    occ_rate = X[:, 2::3].mean()
    print(f"Saved {args.n} samples to {args.out} in {time.time() - t0:.1f}s")
    print(f"  train {len(train_idx)}, val {len(val_idx)}")
    print(f"  labels used: {len(np.unique(y))} / {C.N_LABELS}")
    print(f"  avg occluded features per sample: {occ_rate * 14:.2f} / 14")


if __name__ == "__main__":
    main()
