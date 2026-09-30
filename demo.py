import sys
import csv

sys.path.insert(0, "src")

import numpy as np
import matplotlib.pyplot as plt

from simulator import project, step_dynamics, state_to_pose
from predict import PoseClassifier, load_error_distribution
from particle_filter import RelativePoseParticleFilter


def run_simulation(
    N=1000,
    resample_interval=10,
    alpha=0.9,
    dt=0.1,
    total_time=10.0,
    seed=42
):
    """
    Run one 10-second particle-filter experiment.

    The observation sequence is generated using a dedicated random
    generator so that different PF configurations can use the
    same camera observations for a given seed.
    """

    # --------------------------------------------------
    # Random generators
    # --------------------------------------------------

    # Used ONLY for camera observation noise.
    # This makes observations reproducible and independent
    # of the number of particles.
    observation_rng = np.random.default_rng(seed)

    # Used for PF initialization and leader-control sampling.
    np.random.seed(seed + 1000)

    # --------------------------------------------------
    # Simulation time
    # --------------------------------------------------

    steps = int(total_time / dt)

    time_steps = (np.arange(steps) + 1) * dt

    # --------------------------------------------------
    # Initial true flight state
    # --------------------------------------------------

    # State:
    # [px, py, pz,
    #  phi_L, theta_L, psi_L,
    #  phi_F, theta_F, psi_F]

    true_state = np.array([
        -90.0,
        -20.0,
        -10.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    ])

    # --------------------------------------------------
    # Aircraft controls
    # --------------------------------------------------

    # [vx, theta_cmd, phi_cmd]

    u_follower = np.array([
        25.0,
        0.0,
        0.0
    ])

    u_leader_true = np.array([
        20.0,
        0.0,
        0.0
    ])

    # --------------------------------------------------
    # Load ANN and error distribution
    # --------------------------------------------------

    clf = PoseClassifier()

    deltas = load_error_distribution()

    # --------------------------------------------------
    # Create particle filter
    # --------------------------------------------------

    pf = RelativePoseParticleFilter(
        N=N,
        alpha=alpha
    )

    # --------------------------------------------------
    # Tracking metrics
    # --------------------------------------------------

    ann_pos_errors = []
    ann_att_errors = []

    pf_pos_errors = []
    pf_att_errors = []

    # --------------------------------------------------
    # Main simulation loop
    # --------------------------------------------------

    for t_idx in range(steps):

        # ----------------------------------------------
        # 1. Update true aircraft state
        # ----------------------------------------------

        true_state = step_dynamics(
            true_state,
            u_follower,
            u_leader_true
        )

        true_pose = state_to_pose(true_state)

        # ----------------------------------------------
        # 2. Generate camera observation
        # ----------------------------------------------

        # Same observation sequence can be reproduced
        # for different PF configurations.
        obs = project(
            true_pose,
            noise=True,
            rng=observation_rng
        )

        # ----------------------------------------------
        # 3. ANN coarse pose prediction
        # ----------------------------------------------

        _, coarse_pose = clf.predict(obs)

        # ----------------------------------------------
        # 4. ANN error
        # ----------------------------------------------

        ann_pos_err = np.linalg.norm(
            true_pose[0:3] - coarse_pose[0:3]
        )

        ann_att_err = np.abs(
            true_pose[3] - coarse_pose[3]
        )

        ann_pos_errors.append(ann_pos_err)
        ann_att_errors.append(ann_att_err)

        # ----------------------------------------------
        # 5. Particle filter
        # ----------------------------------------------

        if t_idx == 0:

            # Initialize around ANN estimate using
            # samples from ANN error distribution.
            pf.initialize(
                coarse_pose,
                deltas
            )

        else:

            # Unknown leader controls are treated as noise.
            u_leader_samples = np.random.uniform(
                low=[15.0, -5.0, -5.0],
                high=[25.0, 5.0, 5.0],
                size=(N, 3)
            )

            # Propagate particles.
            pf.propagate(
                u_follower,
                u_leader_samples
            )

        # ----------------------------------------------
        # 6. Weight particles
        # ----------------------------------------------

        pf.weight(obs)

        # ----------------------------------------------
        # 7. Estimate pose BEFORE resampling
        # ----------------------------------------------

        pf_est_state = pf.get_estimate()

        pf_est_pose = state_to_pose(
            pf_est_state
        )

        # ----------------------------------------------
        # 8. Calculate PF error
        # ----------------------------------------------

        pf_pos_err = np.linalg.norm(
            true_pose[0:3] - pf_est_pose[0:3]
        )

        pf_att_err = np.abs(
            true_pose[3] - pf_est_pose[3]
        )

        pf_pos_errors.append(pf_pos_err)
        pf_att_errors.append(pf_att_err)

        # ----------------------------------------------
        # 9. Resample for next iteration
        # ----------------------------------------------

        if t_idx % resample_interval == 0:

            pf.resample(
                coarse_pose,
                deltas
            )

    return (
        time_steps,
        ann_pos_errors,
        ann_att_errors,
        pf_pos_errors,
        pf_att_errors
    )


def main():

    # ==================================================
    # Experiment settings
    # ==================================================

    seeds = [42, 123, 456]

    print("==============================================")
    print("Particle Filter Experiments")
    print("==============================================")

    print("Seeds:", seeds)

    all_results = []

    # Store the first seed's curves for plotting.
    plot_data = None

    # ==================================================
    # Run experiments
    # ==================================================

    for seed in seeds:

        print("\n")
        print("==============================================")
        print(f"SEED {seed}")
        print("==============================================")

        # ------------------------------------------------
        # PF Configuration 1
        # ------------------------------------------------

        (
            t,
            ann_pos,
            ann_att,
            pf1_pos,
            pf1_att
        ) = run_simulation(
            N=1000,
            resample_interval=10,
            alpha=0.9,
            seed=seed
        )

        # ------------------------------------------------
        # PF Configuration 2
        # ------------------------------------------------

        (
            _,
            _,
            _,
            pf2_pos,
            pf2_att
        ) = run_simulation(
            N=5000,
            resample_interval=1,
            alpha=0.9,
            seed=seed
        )

        # ------------------------------------------------
        # Calculate means
        # ------------------------------------------------

        ann_mean_pos = np.mean(ann_pos)
        ann_mean_att = np.mean(ann_att)

        pf1_mean_pos = np.mean(pf1_pos)
        pf1_mean_att = np.mean(pf1_att)

        pf2_mean_pos = np.mean(pf2_pos)
        pf2_mean_att = np.mean(pf2_att)

        # ------------------------------------------------
        # Print seed results
        # ------------------------------------------------

        print("\nANN Alone:")
        print(
            f"Mean position error: "
            f"{ann_mean_pos:.3f} m"
        )
        print(
            f"Mean roll error:     "
            f"{ann_mean_att:.3f} deg"
        )

        print("\nPF Config 1:")
        print(
            f"Mean position error: "
            f"{pf1_mean_pos:.3f} m"
        )
        print(
            f"Mean roll error:     "
            f"{pf1_mean_att:.3f} deg"
        )

        print("\nPF Config 2:")
        print(
            f"Mean position error: "
            f"{pf2_mean_pos:.3f} m"
        )
        print(
            f"Mean roll error:     "
            f"{pf2_mean_att:.3f} deg"
        )

        # ------------------------------------------------
        # Store results
        # ------------------------------------------------

        all_results.append({
            "seed": seed,
            "method": "ANN",
            "particles": 0,
            "resample_interval": 0,
            "mean_position_error_m": ann_mean_pos,
            "mean_roll_error_deg": ann_mean_att
        })

        all_results.append({
            "seed": seed,
            "method": "PF_Config_1",
            "particles": 1000,
            "resample_interval": 10,
            "mean_position_error_m": pf1_mean_pos,
            "mean_roll_error_deg": pf1_mean_att
        })

        all_results.append({
            "seed": seed,
            "method": "PF_Config_2",
            "particles": 5000,
            "resample_interval": 1,
            "mean_position_error_m": pf2_mean_pos,
            "mean_roll_error_deg": pf2_mean_att
        })

        # ------------------------------------------------
        # Save seed 42 curves for the final plot
        # ------------------------------------------------

        if seed == 42:

            plot_data = (
                t,
                ann_pos,
                ann_att,
                pf1_pos,
                pf1_att,
                pf2_pos,
                pf2_att
            )

    # ==================================================
    # Save CSV
    # ==================================================

    csv_path = "results/pf_experiment_results.csv"

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "method",
                "particles",
                "resample_interval",
                "mean_position_error_m",
                "mean_roll_error_deg"
            ]
        )

        writer.writeheader()

        writer.writerows(all_results)

    # ==================================================
    # Overall statistics
    # ==================================================

    print("\n")
    print("==============================================")
    print("OVERALL RESULTS")
    print("==============================================")

    for method in [
        "ANN",
        "PF_Config_1",
        "PF_Config_2"
    ]:

        rows = [
            row
            for row in all_results
            if row["method"] == method
        ]

        position_errors = np.array([
            row["mean_position_error_m"]
            for row in rows
        ])

        roll_errors = np.array([
            row["mean_roll_error_deg"]
            for row in rows
        ])

        position_mean = np.mean(position_errors)
        position_std = np.std(position_errors)

        roll_mean = np.mean(roll_errors)
        roll_std = np.std(roll_errors)

        print(f"\n{method}")

        print(
            f"Position error: "
            f"{position_mean:.3f} +/- "
            f"{position_std:.3f} m"
        )

        print(
            f"Roll error:     "
            f"{roll_mean:.3f} +/- "
            f"{roll_std:.3f} deg"
        )

    # ==================================================
    # Plot results for seed 42
    # ==================================================

    if plot_data is not None:

        (
            t,
            ann_pos,
            ann_att,
            pf1_pos,
            pf1_att,
            pf2_pos,
            pf2_att
        ) = plot_data

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(10, 8),
            sharex=True
        )

        # ------------------------------------------------
        # Position error
        # ------------------------------------------------

        axes[0].plot(
            t,
            ann_pos,
            "k--",
            label="ANN Alone (Coarse)",
            alpha=0.6
        )

        axes[0].plot(
            t,
            pf1_pos,
            "b-",
            label="PF Config 1 (N=1000, interval=10)"
        )

        axes[0].plot(
            t,
            pf2_pos,
            "r-",
            label="PF Config 2 (N=5000, interval=1)"
        )

        axes[0].set_ylabel(
            "Position Error [m]"
        )

        axes[0].set_title(
            "Relative Pose Tracking Performance "
            "(10s Approach, Seed 42)"
        )

        axes[0].grid(True)
        axes[0].legend()

        # ------------------------------------------------
        # Roll error
        # ------------------------------------------------

        axes[1].plot(
            t,
            ann_att,
            "k--",
            label="ANN Alone (Coarse)",
            alpha=0.6
        )

        axes[1].plot(
            t,
            pf1_att,
            "b-",
            label="PF Config 1 (N=1000, interval=10)"
        )

        axes[1].plot(
            t,
            pf2_att,
            "r-",
            label="PF Config 2 (N=5000, interval=1)"
        )

        axes[1].set_xlabel(
            "Time [s]"
        )

        axes[1].set_ylabel(
            "Attitude Roll Error [deg]"
        )

        axes[1].grid(True)
        axes[1].legend()

        plt.tight_layout()

        plt.savefig(
            "results/figures/demo_results.png",
            dpi=300
        )

        plt.close()

    print("\n")
    print("==============================================")
    print("EXPERIMENT COMPLETE")
    print("==============================================")

    print(
        "Detailed results saved to:"
    )

    print(
        "results/pf_experiment_results.csv"
    )

    print(
        "Plot saved to:"
    )

    print(
        "results/figures/demo_results.png"
    )


if __name__ == "__main__":
    main()