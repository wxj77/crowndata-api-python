import numpy as np
from typing import List, Dict
from crowndata_evaluation.services.action_consistency.clustering import define_clusters


class StateSimilarityCalculator:
    """
    A class to compute similarity between trajectories based on epsilon clustering.

    Parameters
    ----------
    epsilon : float
        The coverage distance for clustering.
    """

    def __init__(self, epsilon: float):
        self.epsilon = epsilon  # Coverage distance for clustering

    def compute_similarity(
        self, trajectory: np.ndarray, trajectories: List[np.ndarray]
    ) -> float:
        """
        Compute similarity between one trajectory to multiple trajectories group.

        Parameters
        ----------
        trajectory : np.ndarray
        trajectories : List[np.ndarray]

        Returns:
        -------
        float
            similarity score for single data
        """
        clusters = self._get_clusters(trajectories)

        # Compare each trajectory with the defined clusters
        return self._compute_trajectory_similarity(trajectory, clusters)

    def _compute_trajectory_similarity(
        self, trajectory: np.ndarray, clusters: Dict[int, np.ndarray]
    ) -> float:
        """
        Compute similarity of a trajectory with respect to the clusters.

        Similarity is the ratio of matched states to total states in the trajectory.

        Parameters
        ----------
        trajectory : np.ndarray
            The trajectory to compare.
        clusters : Dict[int, np.ndarray]
            The clusters generated from all trajectories.

        Returns
        -------
        float
            Similarity score between the trajectory and the clusters, in the range [0, 1].
        """
        total_matches = 0
        total_states = len(trajectory)

        for state in trajectory:
            for cluster in clusters.values():
                distances = np.linalg.norm(cluster - state, axis=1)
                if np.any(
                    distances <= self.epsilon
                ):  # State matches if it's within epsilon distance
                    total_matches += 1
                    break

        # Similarity is the ratio of matched states to total states
        similarity = total_matches / total_states if total_states > 0 else 0

        return similarity

    def _get_clusters(self, trajectories: List[np.ndarray]) -> float:
        """
        Compute similarity between one trajectory to multiple trajectories group.

        Parameters
        ----------
        trajectories : List[np.ndarray]

        Returns:
        -------
        clusters : dict
        A dictionary where each key is a cluster label (integer) and the value
        is an array of shape (n_samples_in_cluster, n_features) containing the
        data points in that cluster.
        """
        all_states = np.vstack(trajectories)
        return define_clusters(all_states, self.epsilon)
