"""Simulator for Project 9: camera projection, occlusion, pose bins, and relative dynamics.

CONVENTIONS (read HANDOFF.md for the full picture)
- World frame = leader-level frame: leader at the origin, x forward, y right, z down.
- pose = [x, y, z, phi, theta, psi]
    x, y, z  : follower position relative to leader (m)          -> r_F/L in the paper
    phi      : leader roll relative to follower (deg)             -> the 4th label dimension
    theta,psi: leader pitch / yaw relative to follower (deg)      -> nuisance, not in the label
- Follower attitude is treated as identity (relative-attitude, small-angle simplification).
- Camera sits at the follower and is gimballed to always point at the leader's origin.
- Features: 42 values = (u, v, occluded) for each of the 14 aircraft points.
  Occluded points get u = v = 0 and occluded = 1.

All functions accept a single pose (shape (6,)) or a batch (shape (N, 6)).
"""
import numpy as np

import config as C
from aircraft import POINTS, NORMALS, N_FEATURES


# --------------------------------------------------------------------------- rotations
def rotation_matrix(phi_deg, theta_deg, psi_deg):
    """Body -> world rotation, ZYX (yaw, pitch, roll) convention. Batched over leading dims."""
    phi, theta, psi = (np.deg2rad(np.asarray(a, dtype=float)) for a in (phi_deg, theta_deg, psi_deg))
    cf, sf = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(psi), np.sin(psi)
    R = np.empty(np.shape(phi) + (3, 3))
    R[..., 0, 0] = cp * ct
    R[..., 0, 1] = cp * st * sf - sp * cf
    R[..., 0, 2] = cp * st * cf + sp * sf
    R[..., 1, 0] = sp * ct
    R[..., 1, 1] = sp * st * sf + cp * cf
    R[..., 1, 2] = sp * st * cf - cp * sf
    R[..., 2, 0] = -st
    R[..., 2, 1] = ct * sf
    R[..., 2, 2] = ct * cf
    return R


# --------------------------------------------------------------------------- projection
def _camera_axes(p):
    """Camera axes (right, down, forward) in world frame for cameras at positions p (N, 3)."""
    fwd = -p / np.linalg.norm(p, axis=1, keepdims=True)          # look at the leader (origin)
    world_down = np.array([0.0, 0.0, 1.0])
    right = np.cross(world_down, fwd)
    norm = np.linalg.norm(right, axis=1, keepdims=True)
    degenerate = norm[:, 0] < 1e-6                                # looking straight up/down
    right[degenerate] = [0.0, 1.0, 0.0]
    norm[degenerate] = 1.0
    right = right / norm
    down = np.cross(fwd, right)
    return right, down, fwd


def project(pose, noise=True, rng=None, return_pixels=False):
    """Project the 14 leader features into the follower camera.

    Returns features of shape (42,) or (N, 42): [u0, v0, occ0, u1, v1, occ1, ...].
    noise=True adds Gaussian pixel noise to visible points (use False inside the particle filter).
    return_pixels=True also returns (uv (N,14,2), visible (N,14)) for plotting.
    """
    pose = np.asarray(pose, dtype=float)
    single = pose.ndim == 1
    pose = np.atleast_2d(pose)
    p = pose[:, :3]

    R = rotation_matrix(pose[:, 3], pose[:, 4], pose[:, 5])       # (N, 3, 3)
    pts_w = np.einsum("nij,kj->nki", R, POINTS)                   # (N, 14, 3)
    nrm_w = np.einsum("nij,kj->nki", R, NORMALS)                  # (N, 14, 3)

    right, down, fwd = _camera_axes(p)
    d = pts_w - p[:, None, :]                                     # camera -> point
    xc = np.einsum("nkj,nj->nk", d, right)
    yc = np.einsum("nkj,nj->nk", d, down)
    zc = np.einsum("nkj,nj->nk", d, fwd)

    u = C.FX * xc / zc + C.CX
    v = C.FY * yc / zc + C.CY

    to_cam = -d / np.linalg.norm(d, axis=2, keepdims=True)
    facing = np.einsum("nkj,nkj->nk", nrm_w, to_cam)
    visible = (facing > C.OCCLUSION_THRESHOLD) & (zc > 0)

    if noise and C.PIXEL_NOISE_STD > 0:
        rng = rng or np.random.default_rng()
        u = u + rng.normal(0, C.PIXEL_NOISE_STD, u.shape)
        v = v + rng.normal(0, C.PIXEL_NOISE_STD, v.shape)

    feats = np.zeros((pose.shape[0], N_FEATURES, 3))
    feats[..., 0] = np.where(visible, u, 0.0)
    feats[..., 1] = np.where(visible, v, 0.0)
    feats[..., 2] = (~visible).astype(float)
    feats = feats.reshape(pose.shape[0], N_FEATURES * 3)

    if single:
        feats = feats[0]
    if return_pixels:
        uv = np.stack([u, v], axis=-1)
        return (feats, uv[0], visible[0]) if single else (feats, uv, visible)
    return feats


# --------------------------------------------------------------------------- pose bins
_KEYS = ["x", "y", "z", "phi"]
_N = np.array([C.BINS[k][2] for k in _KEYS])                      # (10, 10, 8, 6)
_LO = np.array([C.BINS[k][0] for k in _KEYS])
_HI = np.array([C.BINS[k][1] for k in _KEYS])
_WIDTH = (_HI - _LO) / _N


def pose_to_label(pose):
    """pose (..., >=4) -> integer label in [0, 4799]. Uses x, y, z, phi only."""
    pose = np.asarray(pose, dtype=float)
    idx = np.floor((pose[..., :4] - _LO) / _WIDTH).astype(int)
    idx = np.clip(idx, 0, _N - 1)
    return np.ravel_multi_index(np.moveaxis(idx, -1, 0), _N)


def label_to_pose(label):
    """label (...) -> bin-centre pose (..., 4) = [x, y, z, phi]."""
    idx = np.stack(np.unravel_index(np.asarray(label), _N), axis=-1)
    return _LO + (idx + 0.5) * _WIDTH


# --------------------------------------------------------------------------- sampling
def sample_poses(n, rng=None):
    """Uniform random poses inside the Table I ranges, rejecting near-collisions."""
    rng = rng or np.random.default_rng()
    out = np.empty((0, 6))
    while len(out) < n:
        m = int((n - len(out)) * 1.2) + 10
        pos = rng.uniform(_LO[:3], _HI[:3], size=(m, 3))
        phi = rng.uniform(_LO[3], _HI[3], size=(m, 1))
        theta = rng.uniform(-C.NUISANCE_PITCH_DEG, C.NUISANCE_PITCH_DEG, size=(m, 1))
        psi = rng.uniform(-C.NUISANCE_YAW_DEG, C.NUISANCE_YAW_DEG, size=(m, 1))
        batch = np.hstack([pos, phi, theta, psi])
        batch = batch[np.linalg.norm(pos, axis=1) >= C.MIN_RANGE_M]
        out = np.vstack([out, batch])
    return out[:n]


# --------------------------------------------------------------------------- dynamics
# state = [px, py, pz, phi_L, theta_L, psi_L, phi_F, theta_F, psi_F]   (m, deg)
#   p = follower position relative to leader. Batched over leading dims.
# control u = [vx (m/s), theta_cmd (deg), phi_cmd (deg)] for each aircraft.

def _velocity(vx, theta_deg, phi_deg):
    """Simplified kinematics: roll -> sideways speed, pitch -> climb (z is down)."""
    lateral = C.K_LATERAL * np.sin(np.deg2rad(phi_deg))
    vertical = -C.K_VERTICAL * np.sin(np.deg2rad(theta_deg))
    return np.stack(np.broadcast_arrays(vx, lateral, vertical), axis=-1)


def step_dynamics(state, u_follower, u_leader, dt=C.DT):
    """Advance the relative state by dt. Works on one state (9,) or many (N, 9)."""
    state = np.asarray(state, dtype=float)
    uF = np.asarray(u_follower, dtype=float)
    uL = np.asarray(u_leader, dtype=float)
    new = state.copy()
    a = dt / C.ATTITUDE_TAU

    # attitude: first-order lag toward commanded roll/pitch, yaw held
    new[..., 3] = state[..., 3] + a * (uL[..., 2] - state[..., 3])   # phi_L
    new[..., 4] = state[..., 4] + a * (uL[..., 1] - state[..., 4])   # theta_L
    new[..., 6] = state[..., 6] + a * (uF[..., 2] - state[..., 6])   # phi_F
    new[..., 7] = state[..., 7] + a * (uF[..., 1] - state[..., 7])   # theta_F

    vL = _velocity(uL[..., 0], new[..., 4], new[..., 3])
    vF = _velocity(uF[..., 0], new[..., 7], new[..., 6])
    new[..., 0:3] = state[..., 0:3] + (vF - vL) * dt
    return new


def state_to_pose(state):
    """Relative state (..., 9) -> pose (..., 6) that project() understands."""
    state = np.asarray(state, dtype=float)
    pose = np.empty(state.shape[:-1] + (6,))
    pose[..., 0:3] = state[..., 0:3]
    pose[..., 3:6] = state[..., 3:6] - state[..., 6:9]
    return pose
