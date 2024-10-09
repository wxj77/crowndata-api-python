from typing import Tuple
import numpy as np
from sklearn.cluster import KMeans
from crowndata_evaluation.services.utils import cosine_similarity


class TrajectorySimilarity:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters

    def dual_action_variance(
        self, states: np.ndarray, actions: np.ndarray, epsilon: float
    ) -> float:
        """
        Compute the variance in actions based on the nearby states defined by epsilon.

        Args:
            states (np.ndarray): Array of states
            actions (np.ndarray): Array of actions
            epsilon (float): Distance threshold to consider states as neighbors

        Returns:
            float: The average variance of actions in nearby states
        """
        N = len(states)
        variances = []

        for i in range(N):
            state = states[i]
            action = actions[i]

            # Calculate distances from the current state to all other states
            distances = np.linalg.norm(states - state, axis=1)
            nearby_indices = np.where(distances <= epsilon)[0]

            if len(nearby_indices) > 0:
                # Compute variance of actions for the nearby states
                cluster_actions = actions[nearby_indices]
                cluster_mean = np.mean(cluster_actions, axis=0)
                cluster_variance = np.mean((cluster_actions - cluster_mean) ** 2)
                variances.append(cluster_variance)

        # Return the mean variance across all states
        action_variance = np.mean(variances) if variances else 0
        return action_variance

    def dual_state_similarity(
        self, traj_a: np.ndarray, traj_b: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute the similarity between two trajectories using KMeans clustering.

        Args:
            traj_a (np.ndarray): Trajectory A
            traj_b (np.ndarray): Trajectory B

        Returns:
            float: Similarity between the two trajectories
        """
        # Combine both trajectories into one dataset for clustering
        combined_data = np.vstack((traj_a, traj_b))

        # Perform KMeans clustering on the combined data with random seed 42
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        kmeans.fit(combined_data)

        # Get cluster labels for each trajectory
        labels_a = kmeans.predict(traj_a)
        labels_b = kmeans.predict(traj_b)

        # Compute normalized histograms (cluster distributions) for both trajectories
        hist_a = np.bincount(labels_a, minlength=self.n_clusters) / len(traj_a)
        hist_b = np.bincount(labels_b, minlength=self.n_clusters) / len(traj_b)

        # Compute similarity as the inverse of the norm between the two histograms
        similarity_score = 1 - np.linalg.norm(hist_a - hist_b)
        cosine_similarity_score = cosine_similarity(hist_a, hist_b)

        return similarity_score, cosine_similarity_score
