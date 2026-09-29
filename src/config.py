"""Central config: every number the project depends on lives here."""
import numpy as np

SEED = 42

# ---- Pose bins (paper Table I) ----
# Relative position of the FOLLOWER w.r.t. the LEADER, in metres, in the leader-level frame
# (x forward, y right, z down). Roll = leader roll relative to follower, in degrees.
BINS = {
    "x":   (-100.0, 50.0, 10),
    "y":   (-50.0,  50.0, 10),
    "z":   (-20.0,  20.0, 8),
    "phi": (-45.0,  45.0, 6),
}
N_LABELS = 10 * 10 * 8 * 6  # 4800

# Nuisance rotations the classifier ignores (paper drops relative pitch/yaw from the labels)
NUISANCE_PITCH_DEG = 5.0
NUISANCE_YAW_DEG = 5.0

# Reject poses where the two aircraft would practically collide
MIN_RANGE_M = 15.0

# ---- Camera (pinhole, gimballed to always point at the leader) ----
IMG_W, IMG_H = 1280, 960
FOV_DEG = 40.0
FX = (IMG_W / 2) / np.tan(np.deg2rad(FOV_DEG / 2))
FY = FX
CX, CY = IMG_W / 2, IMG_H / 2

# Detector noise on visible feature pixels (std, px). Stand-in for an imperfect CNN detector.
PIXEL_NOISE_STD = 1.0

# Occlusion rule: a feature is visible if dot(outward normal, direction to camera) > this
OCCLUSION_THRESHOLD = -0.3

# ---- Dynamics ----
DT = 0.1                 # s, simulation timestep
ATTITUDE_TAU = 0.5       # s, first-order lag from commanded to actual pitch/roll
K_LATERAL = 30.0         # m/s of sideways speed per unit sin(roll)
K_VERTICAL = 30.0        # m/s of climb per unit sin(pitch)
