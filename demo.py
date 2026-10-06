"""
Project 9 demo: ANN coarse pose + Particle Filter refinement.

Runs the 10-second formation approach, compares the ANN alone against two
particle filter configurations, and saves the figures and a CSV of results.

Usage (from anywhere):
    python demo.py              # full experiment, 3 seeds, opens the figures
    python demo.py --quick      # seed 42 only (faster, for a live demo)
    python demo.py --no-show    # save figures without opening windows
"""
import argparse
import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

import numpy as np
import matplotlib.pyplot as plt

import config as C
from aircraft import EDGES
from simulator import project, step_dynamics, state_to_pose
from predict import PoseClassifier, load_error_distribution
from particle_filter import RelativePoseParticleFilter

FIG_DIR = os.path.join(ROOT, "results", "figures")
CSV_PATH = os.path.join(ROOT, "results", "pf_experiment_results.csv")
LINE = "=" * 60


def run_simulation(N=1000, resample_interval=10, alpha=0.9, dt=0.1,
                   total_time=10.0, seed=42, record=False):
    """
    Run one 10-second particle-filter experiment.

    Observation noise uses its own generator, so every PF configuration sees
    the exact same camera observations for a given seed.
    With record=True, the true, ANN and PF poses at every step are also returned
    (used only for the camera-view figure; results are unchanged).
    """
    observation_rng = np.random.default_rng(seed)
    np.random.seed(seed + 1000)

    steps = int(total_time / dt)
    time_steps = (np.arange(steps) + 1) * dt

    # State: [px, py, pz, phi_L, theta_L, psi_L, phi_F, theta_F, psi_F]
    # Follower starts 90 m behind, 20 m left and 10 m above the leader.
    true_state = np.array([-90.0, -20.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Controls: [vx (m/s), theta_cmd (deg), phi_cmd (deg)]
    u_follower = np.array([25.0, 0.0, 0.0])      # follower is 5 m/s faster, so it closes in
    u_leader_true = np.array([20.0, 0.0, 0.0])

    clf = PoseClassifier()
    deltas = load_error_distribution()
    pf = RelativePoseParticleFilter(N=N, alpha=alpha)

    ann_pos_errors, ann_att_errors = [], []
    pf_pos_errors, pf_att_errors = [], []
    trace = {"true": [], "ann": [], "pf": [], "obs": []}

    for t_idx in range(steps):
        # 1. True aircraft motion
        true_state = step_dynamics(true_state, u_follower, u_leader_true)
        true_pose = state_to_pose(true_state)

        # 2. Camera observation (14 points with pixel noise)
        obs = project(true_pose, noise=True, rng=observation_rng)

        # 3. ANN coarse estimate (bin centre)
        _, coarse_pose = clf.predict(obs)
        ann_pos_errors.append(np.linalg.norm(true_pose[0:3] - coarse_pose[0:3]))
        ann_att_errors.append(np.abs(true_pose[3] - coarse_pose[3]))

        # 4. Particle filter: initialise once, then propagate
        if t_idx == 0:
            pf.initialize(coarse_pose, deltas)
        else:
            # Leader controls are unknown to the follower, so they are sampled as noise
            u_leader_samples = np.random.uniform(low=[15.0, -5.0, -5.0],
                                                 high=[25.0, 5.0, 5.0], size=(N, 3))
            pf.propagate(u_follower, u_leader_samples)

        # 5. Weight against the observation, then estimate
        pf.weight(obs)
        pf_est_pose = state_to_pose(pf.get_estimate())
        pf_pos_errors.append(np.linalg.norm(true_pose[0:3] - pf_est_pose[0:3]))
        pf_att_errors.append(np.abs(true_pose[3] - pf_est_pose[3]))

        if record:
            trace["true"].append(true_pose.copy())
            trace["ann"].append(np.r_[coarse_pose, 0.0, 0.0])
            trace["pf"].append(pf_est_pose.copy())
            trace["obs"].append(obs.copy())

        # 6. Resample for the next step
        if t_idx % resample_interval == 0:
            pf.resample(coarse_pose, deltas)

    out = (time_steps, ann_pos_errors, ann_att_errors, pf_pos_errors, pf_att_errors)
    return out + ({k: np.array(v) for k, v in trace.items()},) if record else out


def plot_errors(t, ann_pos, ann_att, pf1_pos, pf1_att, pf2_pos, pf2_att):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, ann, pf1, pf2, ylabel in [
        (axes[0], ann_pos, pf1_pos, pf2_pos, "Position Error [m]"),
        (axes[1], ann_att, pf1_att, pf2_att, "Attitude Roll Error [deg]"),
    ]:
        ax.plot(t, ann, "k--", label="ANN Alone (Coarse)", alpha=0.6)
        ax.plot(t, pf1, "b-", label="PF Config 1 (N=1000, interval=10)")
        ax.plot(t, pf2, "r-", label="PF Config 2 (N=5000, interval=1)")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()
    axes[0].set_title("Relative Pose Tracking Performance (10s Approach, Seed 42)")
    axes[1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "demo_results.png"), dpi=300)
    return fig


def plot_camera_views(trace, times=(0.1, 5.0, 10.0), dt=0.1):
    """What the follower camera sees, with the ANN and PF estimates projected on top."""
    fig, axes = plt.subplots(1, len(times), figsize=(5.2 * len(times), 4.6))
    for ax, ts in zip(axes, times):
        k = int(round(ts / dt)) - 1
        _, uv_true, vis = project(trace["true"][k], noise=False, return_pixels=True)
        _, uv_ann, _ = project(trace["ann"][k], noise=False, return_pixels=True)
        _, uv_pf, _ = project(trace["pf"][k], noise=False, return_pixels=True)

        for a, b in EDGES:
            ax.plot(uv_true[[a, b], 0], uv_true[[a, b], 1], color="#bbb", lw=0.8)
        obs = trace["obs"][k].reshape(14, 3)
        seen = obs[:, 2] == 0
        ax.scatter(obs[seen, 0], obs[seen, 1], s=40, color="#d62728", zorder=3,
                   label="Observed points")
        ax.scatter(uv_ann[vis, 0], uv_ann[vis, 1], s=40, marker="x", color="#555",
                   zorder=4, label="ANN estimate")
        ax.scatter(uv_pf[vis, 0], uv_pf[vis, 1], s=70, facecolors="none",
                   edgecolors="#1f5fbf", lw=1.4, zorder=5, label="PF estimate")

        tp, ap, pp = trace["true"][k], trace["ann"][k], trace["pf"][k]
        ax.set_title(
            f"t = {ts:.1f} s\n"
            f"ANN error {np.linalg.norm(tp[:3] - ap[:3]):.2f} m, "
            f"PF error {np.linalg.norm(tp[:3] - pp[:3]):.2f} m", fontsize=10)
        pts = np.vstack([uv_true[vis], uv_ann[vis], uv_pf[vis]])
        cx, cy = pts.mean(axis=0)
        half = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]) * 1.3, 60) * 0.75
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy + half * 0.8, cy - half * 0.8)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Follower camera view, zoomed on the leader (seed 42, PF Config 2)", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIG_DIR, "demo_camera_views.png"), dpi=200)
    return fig


def summarise(rows, method):
    sel = [r for r in rows if r["method"] == method]
    pos = np.array([r["mean_position_error_m"] for r in sel])
    roll = np.array([r["mean_roll_error_deg"] for r in sel])
    return pos.mean(), pos.std(), roll.mean(), roll.std()


def main():
    parser = argparse.ArgumentParser(description="ANN + Particle Filter pose estimation demo")
    parser.add_argument("--quick", action="store_true", help="run seed 42 only")
    parser.add_argument("--no-show", action="store_true", help="save figures without opening them")
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    seeds = [42] if args.quick else [42, 123, 456]

    print(LINE)
    print("Vision-Based Pose Estimation for Formation Flying (Team 9)")
    print(LINE)
    print("Pipeline : 14 points -> camera projection -> 42 features")
    print("           -> ANN (4800 pose classes) -> Particle Filter")
    print("Scenario : 10 s approach, follower closes from 90 m behind the leader")
    print(f"Seeds    : {seeds}")

    started = time.time()
    rows, plot_data, trace = [], None, None

    for seed in seeds:
        print(f"\n{LINE}\nSEED {seed}\n{LINE}")
        t, ann_pos, ann_att, pf1_pos, pf1_att = run_simulation(
            N=1000, resample_interval=10, alpha=0.9, seed=seed)
        _, _, _, pf2_pos, pf2_att, tr = run_simulation(
            N=5000, resample_interval=1, alpha=0.9, seed=seed, record=True)

        for name, pos, att, n, ri in [
            ("ANN", ann_pos, ann_att, 0, 0),
            ("PF_Config_1", pf1_pos, pf1_att, 1000, 10),
            ("PF_Config_2", pf2_pos, pf2_att, 5000, 1),
        ]:
            print(f"{name:<12} position {np.mean(pos):7.3f} m    roll {np.mean(att):6.3f} deg")
            rows.append({"seed": seed, "method": name, "particles": n,
                         "resample_interval": ri,
                         "mean_position_error_m": float(np.mean(pos)),
                         "mean_roll_error_deg": float(np.mean(att))})

        if seed == 42:
            plot_data = (t, ann_pos, ann_att, pf1_pos, pf1_att, pf2_pos, pf2_att)
            trace = tr

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{LINE}\nOVERALL RESULTS (mean +/- std over {len(seeds)} seed(s))\n{LINE}")
    ann = summarise(rows, "ANN")
    for method in ["ANN", "PF_Config_1", "PF_Config_2"]:
        p, ps, r, rs = summarise(rows, method)
        print(f"{method:<12} position {p:6.3f} +/- {ps:.3f} m    roll {r:6.3f} +/- {rs:.3f} deg")
    p2 = summarise(rows, "PF_Config_2")
    print(f"\nPF Config 2 cuts position error by {100 * (1 - p2[0] / ann[0]):.1f}% "
          f"and roll error by {100 * (1 - p2[2] / ann[2]):.1f}% vs the ANN alone.")

    plot_errors(*plot_data)
    plot_camera_views(trace)

    print(f"\n{LINE}\nDONE in {time.time() - started:.1f} s\n{LINE}")
    print("Results CSV : results/pf_experiment_results.csv")
    print("Figures     : results/figures/demo_results.png")
    print("              results/figures/demo_camera_views.png")

    if not args.no_show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
