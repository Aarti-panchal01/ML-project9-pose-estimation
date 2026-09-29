"""The 14 known structural features on the leader aircraft (fighter-like, ~15 m long, 10 m span).

Body frame: x forward (nose), y right wing, z down. Units: metres.
Each feature has an outward normal, used by the occlusion rule in simulator.py.
"""
import numpy as np

FEATURES = [
    # name,               position (x, y, z),   outward normal
    ("nose",              ( 7.5,  0.0,  0.0),   ( 1.0,  0.0,  0.0)),
    ("cockpit_top",       ( 4.5,  0.0, -1.2),   ( 0.3,  0.0, -1.0)),
    ("left_wingtip",      (-1.5, -5.0,  0.0),   ( 0.0, -1.0,  0.0)),
    ("right_wingtip",     (-1.5,  5.0,  0.0),   ( 0.0,  1.0,  0.0)),
    ("left_wing_top",     (-0.5, -2.8, -0.1),   ( 0.0,  0.0, -1.0)),
    ("right_wing_top",    (-0.5,  2.8, -0.1),   ( 0.0,  0.0, -1.0)),
    ("left_wing_pylon",   (-0.5, -2.8,  0.4),   ( 0.0,  0.0,  1.0)),
    ("right_wing_pylon",  (-0.5,  2.8,  0.4),   ( 0.0,  0.0,  1.0)),
    ("tail_fin_tip",      (-6.5,  0.0, -3.0),   (-0.3,  0.0, -1.0)),
    ("left_stab_tip",     (-7.0, -2.8,  0.0),   ( 0.0, -1.0,  0.0)),
    ("right_stab_tip",    (-7.0,  2.8,  0.0),   ( 0.0,  1.0,  0.0)),
    ("exhaust",           (-7.5,  0.0,  0.0),   (-1.0,  0.0,  0.0)),
    ("intake",            ( 3.0,  0.0,  1.0),   ( 0.3,  0.0,  1.0)),
    ("belly_aft",         (-3.0,  0.0,  0.8),   ( 0.0,  0.0,  1.0)),
]

N_FEATURES = len(FEATURES)  # 14
NAMES = [f[0] for f in FEATURES]
POINTS = np.array([f[1] for f in FEATURES], dtype=float)                   # (14, 3)
_normals = np.array([f[2] for f in FEATURES], dtype=float)
NORMALS = _normals / np.linalg.norm(_normals, axis=1, keepdims=True)       # (14, 3)

# Line segments between features, only used to draw a wireframe in plots / the demo
EDGES = [
    (0, 1), (1, 8), (0, 12), (12, 13), (13, 11), (8, 11),
    (2, 4), (4, 5), (5, 3), (2, 6), (6, 7), (7, 3),
    (9, 11), (11, 10),
]
