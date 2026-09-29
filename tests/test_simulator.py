"""Quick sanity checks. Run: python tests/test_simulator.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from simulator import (project, pose_to_label, label_to_pose, sample_poses,  # noqa: E402
                       step_dynamics, state_to_pose)
from aircraft import NAMES  # noqa: E402


def test_label_roundtrip():
    poses = sample_poses(5000, np.random.default_rng(0))
    labels = pose_to_label(poses)
    assert labels.min() >= 0 and labels.max() < 4800
    assert (pose_to_label(label_to_pose(labels)) == labels).all()


def test_batch_matches_single():
    poses = sample_poses(5, np.random.default_rng(1))
    batch = project(poses, noise=False)
    single = np.stack([project(p, noise=False) for p in poses])
    assert batch.shape == (5, 42) and np.allclose(batch, single)


def test_view_from_behind():
    _, uv, vis = project([-30, 0, 0, 0, 0, 0], noise=False, return_pixels=True)
    visible = {n for n, v in zip(NAMES, vis) if v}
    assert "exhaust" in visible and "nose" not in visible
    tail, belly = NAMES.index("tail_fin_tip"), NAMES.index("belly_aft")
    assert uv[tail, 1] < uv[belly, 1]            # tail fin appears above the belly


def test_view_from_above_hides_belly():
    _, _, vis = project([-30, 0, -10, 0, 0, 0], noise=False, return_pixels=True)
    assert not vis[NAMES.index("belly_aft")]


def test_occluded_features_are_zeroed():
    f = project([-30, 0, 0, 0, 0, 0], noise=False).reshape(14, 3)
    occ = f[:, 2] == 1
    assert occ.any() and np.all(f[occ, :2] == 0)


def test_dynamics_closes_gap():
    s = np.array([-50.0, 5, 0, 0, 0, 0, 0, 0, 0])
    for _ in range(100):                          # 10 s
        s = step_dynamics(s, u_follower=[105, 0, 0], u_leader=[100, 0, 0])
    assert abs(s[0]) < 1e-6                       # 5 m/s faster for 10 s closes 50 m
    assert state_to_pose(s).shape == (6,)


def test_dynamics_batched():
    s = np.zeros((1000, 9))
    s[:, 0] = -50
    out = step_dynamics(s, [100, 0, 0], np.tile([100, 0, 10], (1000, 1)))
    assert out.shape == (1000, 9)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS  {name}")
