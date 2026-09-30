import sys
sys.path.insert(0, "src")

import numpy as np
import matplotlib.pyplot as plt
from simulator import project, step_dynamics, state_to_pose
from predict import PoseClassifier, load_error_distribution
from particle_filter import RelativePoseParticleFilter

def run_simulation(N=1000, resample_interval=10, alpha=0.9, dt=0.1, total_time=10.0):
    steps = int(total_time / dt)
    time_steps = np.linspace(0, total_time, steps)

    # 1. Initialize True Flight Trajectory (Follower closing in on Leader)
    # Initial state: [px, py, pz, phi_L, theta_L, psi_L, phi_F, theta_F, psi_F]
    true_state = np.array([-90.0, -20.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Aircraft controls: [vx (m/s), theta_cmd (deg), phi_cmd (deg)]
    u_follower = np.array([25.0, 0.0, 0.0])
    u_leader_true = np.array([20.0, 0.0, 0.0])

    # Load trained model and error distribution
    clf = PoseClassifier()
    deltas = load_error_distribution()

    # Instantiate Particle Filter
    pf = RelativePoseParticleFilter(N=N, alpha=alpha)

    # Tracking metrics
    ann_pos_errors, ann_att_errors = [], []
    pf_pos_errors, pf_att_errors = [], []

    for t_idx in range(steps):
        # Step dynamics for true trajectory
        true_state = step_dynamics(true_state, u_follower, u_leader_true)
        true_pose = state_to_pose(true_state)

        # Generate camera observation with pixel noise
        obs = project(true_pose, noise=True)

        # Step A: ANN Classifier guess
        _, coarse_pose = clf.predict(obs)

        # Calculate ANN Error
        ann_pos_err = np.linalg.norm(true_pose[0:3] - coarse_pose[0:3])
        ann_att_err = np.abs(true_pose[3] - coarse_pose[3])
        ann_pos_errors.append(ann_pos_err)
        ann_att_errors.append(ann_att_err)

        # Step B: Particle Filter
        if t_idx == 0:
            pf.initialize(coarse_pose, deltas)
        else:
            # Sample unknown leader controls as noise for propagation
            u_leader_samples = np.random.uniform(
                low=[15.0, -5.0, -5.0],
                high=[25.0, 5.0, 5.0],
                size=(N, 3)
            )
            pf.propagate(u_follower, u_leader_samples)

        # Calculate weights based on visible feature pixel errors
        pf.weight(obs)

        # Resample conditionally based on interval parameter
        if t_idx % resample_interval == 0:
            pf.resample(coarse_pose, deltas)

        # Estimate pose from particle distribution
        pf_est_state = pf.get_estimate()
        pf_est_pose = state_to_pose(pf_est_state)

        # Calculate PF Error
        pf_pos_err = np.linalg.norm(true_pose[0:3] - pf_est_pose[0:3])
        pf_att_err = np.abs(true_pose[3] - pf_est_pose[3])
        pf_pos_errors.append(pf_pos_err)
        pf_att_errors.append(pf_att_err)

    return time_steps, ann_pos_errors, ann_att_errors, pf_pos_errors, pf_att_errors


def main():
    print("Running Particle Filter Experiments...")

    # Config 1: N=1000, alpha=0.9 every 10 cycles
    t, ann_pos, ann_att, pf1_pos, pf1_att = run_simulation(N=1000, resample_interval=10, alpha=0.9)

    # Config 2: N=5000, alpha=0.9 every 1 cycle
    _, _, _, pf2_pos, pf2_att = run_simulation(N=5000, resample_interval=1, alpha=0.9)

    # Plotting Results (Replicating Figures 9 and 10 from the paper)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Position Error Plot
    axes[0].plot(t, ann_pos, 'k--', label='ANN Alone (Coarse)', alpha=0.6)
    axes[0].plot(t, pf1_pos, 'b-', label='PF Config 1 (N=1000, interval=10)')
    axes[0].plot(t, pf2_pos, 'r-', label='PF Config 2 (N=5000, interval=1)')
    axes[0].set_ylabel('Position Error [m]')
    axes[0].set_title('Relative Pose Tracking Performance (10s Approach)')
    axes[0].grid(True)
    axes[0].legend()

    # Attitude Error Plot
    axes[1].plot(t, ann_att, 'k--', label='ANN Alone (Coarse)', alpha=0.6)
    axes[1].plot(t, pf1_att, 'b-', label='PF Config 1 (N=1000, interval=10)')
    axes[1].plot(t, pf2_att, 'r-', label='PF Config 2 (N=5000, interval=1)')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Attitude Roll Error [deg]')
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('results/figures/demo_results.png', dpi=300)
    print("Simulation complete! Results saved to 'results/figures/demo_results.png'.")
    plt.show()

if __name__ == '__main__':
    main()
