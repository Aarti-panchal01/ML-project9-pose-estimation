import numpy as np
from simulator import project, step_dynamics, state_to_pose

class RelativePoseParticleFilter:
    def __init__(self, N, alpha=0.9):
        self.N = N
        self.alpha = alpha
        self.particles = None
        self.weights = None

    def initialize(self, coarse_pose, deltas):
        # Step 1: Init (Sample N particles = ANN guess + random draws from error distribution)
        d = deltas[np.random.randint(len(deltas), size=self.N)]
        particle_xyzphi = coarse_pose + d
        
        # Build full (N, 9) state vectors
        self.particles = np.zeros((self.N, 9))
        self.particles[:, 0:3] = particle_xyzphi[:, 0:3]  # x, y, z
        self.particles[:, 3] = particle_xyzphi[:, 3]      # phi_L (roll)
        
        self.weights = np.ones(self.N) / self.N

    def propagate(self, u_follower, u_leader_samples):
        # Step 2: Propagate (Run through dynamics with noisy leader controls)
        self.particles = step_dynamics(self.particles, u_follower, u_leader_samples)

    def weight(self, obs):
        # Step 3: Weight (Project particles and score vs observed points via 1/MSE)
        pred = project(state_to_pose(self.particles), noise=False)
        
        # Find which points are visible in the camera image (occlusion flag == 0)
        visible_flags = (obs[2::3] == 0)
        
        # Create a mask for the (N, 42) array to select only the u,v coords of visible points
        mask = np.zeros(42, dtype=bool)
        mask[0::3] = visible_flags  # u coordinates
        mask[1::3] = visible_flags  # v coordinates
        
        # If all points are occluded (rare), keep weights uniform
        if not np.any(mask):
            self.weights = np.ones(self.N) / self.N
            return
            
        # Calculate MSE only on the visible pixel coordinates
        diff = pred[:, mask] - obs[mask]
        mse = np.mean(diff**2, axis=1)
        
        # Weight = 1 / MSE. Add 1e-6 to avoid divide-by-zero if a particle matches perfectly
        self.weights = 1.0 / (mse + 1e-6)
        
        # Normalize weights so they sum to 1
        self.weights /= np.sum(self.weights)

    def resample(self, coarse_pose, deltas):
        # Step 4: Resample (Keep floor(alpha * N) weighted particles, rest are fresh)
        num_keep = int(self.alpha * self.N)
        num_fresh = self.N - num_keep
        
        # Draw from current particles using their weights
        keep_indices = np.random.choice(self.N, size=num_keep, p=self.weights)
        kept_particles = self.particles[keep_indices]
        
        # Draw fresh particles from the ANN guess + error distribution
        d = deltas[np.random.randint(len(deltas), size=num_fresh)]
        fresh_xyzphi = coarse_pose + d
        fresh_particles = np.zeros((num_fresh, 9))
        fresh_particles[:, 0:3] = fresh_xyzphi[:, 0:3]
        fresh_particles[:, 3] = fresh_xyzphi[:, 3]
        
        # Combine into the new particle set and reset weights
        self.particles = np.vstack((kept_particles, fresh_particles))
        self.weights = np.ones(self.N) / self.N

    def get_estimate(self):
        # Return the weighted average state across all particles
        return np.average(self.particles, axis=0, weights=self.weights)
