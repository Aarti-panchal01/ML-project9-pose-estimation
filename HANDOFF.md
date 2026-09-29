# Handoff: Person 1 -> Person 2

Everything you need to build the particle filter. Nothing here should require asking Person 1.

## Coordinate frames and units

- **World frame** = leader-level frame. Leader sits at the origin. x forward, y right, z **down**.
- Distances are in **metres**, angles are in **degrees** everywhere.
- The camera is on the follower and is gimballed to always point at the leader's origin.
- Follower attitude is treated as identity. Only *relative* attitude matters for the image
  (a small-angle simplification; see "Deviations from the paper" below).

## The pose vector (6 values)

```
pose = [x, y, z, phi, theta, psi]
         |  |  |   |     |     `- leader yaw   relative to follower (nuisance, +-5 deg in data)
         |  |  |   |     `------- leader pitch relative to follower (nuisance, +-5 deg in data)
         |  |  |   `------------- leader roll  relative to follower (the 4th label dimension)
         `--`--`----------------- follower position relative to leader (r_F/L), metres
```

The **label** only uses `[x, y, z, phi]`, matching paper Table I.

## The state vector (9 values), used for dynamics

```
state = [px, py, pz,  phi_L, theta_L, psi_L,  phi_F, theta_F, psi_F]
```

`state_to_pose(state)` turns it into the 6-value pose that `project()` takes.

Control per aircraft: `u = [vx (m/s), theta_cmd (deg), phi_cmd (deg)]`.
Roll produces sideways speed and pitch produces climb (`K_LATERAL`, `K_VERTICAL` in config).
Attitude follows the command with a first-order lag (`ATTITUDE_TAU = 0.5 s`). `DT = 0.1 s`.

## The feature vector (42 values)

```
[u0, v0, occ0,  u1, v1, occ1,  ...,  u13, v13, occ13]
```

- `u, v` = pixel coordinates in a 1280 x 960 image. `occ` = 1 if hidden, else 0.
- Hidden points have `u = v = 0`.
- Feature order = `aircraft.NAMES` (nose, cockpit_top, left_wingtip, ...).

## Functions you'll call

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from simulator import project, step_dynamics, state_to_pose, pose_to_label, label_to_pose
from predict import PoseClassifier, load_error_distribution

clf = PoseClassifier()                       # loads models/pose_ann.joblib
deltas = load_error_distribution()           # (M, 4) of (true - classified) for [x, y, z, phi]

# observation at one timestep
obs = project(state_to_pose(true_state), noise=True)       # (42,)
label, coarse = clf.predict(obs)                           # coarse = [x, y, z, phi] bin centre

# init / resample particles around the ANN guess (paper step 1 and step 4)
d = deltas[np.random.randint(len(deltas), size=N)]         # (N, 4)
particle_xyzphi = coarse + d

# build full (N, 9) states from it: follower attitude 0, pitch/yaw unknown -> 0 (or small noise)
particles = np.zeros((N, 9))
particles[:, 0:3] = particle_xyzphi[:, 0:3]
particles[:, 3] = particle_xyzphi[:, 3]                    # phi_L (relative roll, since phi_F = 0)

# propagate all particles at once (paper step 2); leader controls unknown -> sample them
particles = step_dynamics(particles, u_follower, u_leader_samples)   # (N, 9) -> (N, 9)

# predicted observations for every particle (paper step 3): NO noise here
pred = project(state_to_pose(particles), noise=False)      # (N, 42)
```

`project()`, `step_dynamics()` and `state_to_pose()` all work on one row or on (N, ...) batches.
For the likelihood, compare only features visible in `obs` (where `obs[2::3] == 0`), as the paper does.

## Results from Part 1

Validation set: **29.0% exact bin, 75.1% top-5, 87.1% within one bin in every dimension.**
Median abs error of the coarse estimate: 3.8 m (x), 3.0 m (y), 2.8 m (z), 4.3 deg (roll).
So the particle filter's job is to beat roughly 3-4 m / 4 deg. Full numbers in `results/metrics.json`,
figures in `results/figures/`.

## Deviations from the paper (mention these in the write-up)

1. **No ray-traced render.** The paper renders a full aircraft model. We project 14 points
   directly, which is all the pipeline uses.
2. **Occlusion by surface normals.** A point is hidden when its outward normal faces away from
   the camera (threshold in `config.OCCLUSION_THRESHOLD`). The paper doesn't specify its method.
3. **Pixel noise.** Visible points get 1 px Gaussian noise, standing in for an imperfect CNN
   detector. The paper assumes perfect detection. Set `PIXEL_NOISE_STD = 0` to match it.
4. **Dynamics.** The paper says pitch and roll induce forward and lateral velocities but gives no
   equations. We use roll -> lateral speed and pitch -> vertical speed, which is the physical
   behaviour of an aircraft.
5. **Collision filter.** Poses closer than 15 m are rejected, so 27 of the 4800 bins (those at the
   leader's position) never appear in the data.
6. **Framework.** scikit-learn `MLPClassifier` instead of a deep-learning framework. Same network:
   42 -> 100 -> 100 -> 4800 softmax, ReLU, Adam.
