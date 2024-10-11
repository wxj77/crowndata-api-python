import numpy as np
from typing import Dict, Union
from clustering import define_clusters


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
        self, trajectories: Union[Dict[str, np.ndarray], np.ndarray]
    ) -> float:
        """
        Compute similarity among multiple trajectories or two trajectories based on epsilon clustering.

        Parameters
        ----------
        trajectories : Union[Dict[str, np.ndarray], np.ndarray]
            A dictionary where keys are demo names and values are trajectories, or a single trajectory.

        Returns
        -------
        float
            Global similarity score across all trajectories or the similarity for a single trajectory.
        """
        if isinstance(trajectories, np.ndarray):  # Single trajectory case
            clusters = define_clusters(trajectories, self.epsilon)
            return self._compute_trajectory_similarity(trajectories, clusters)
        else:  # Multiple trajectories case (when passed as a dictionary)
            all_states = np.vstack(list(trajectories.values()))
            clusters = define_clusters(all_states, self.epsilon)

            # Compare each trajectory with the defined clusters
            similarities = {}
            for name, traj in trajectories.items():
                similarity = self._compute_trajectory_similarity(traj, clusters)
                similarities[name] = similarity

            # Global similarity: average similarity across all trajectories
            global_similarity = (
                np.mean(list(similarities.values())) if similarities else 0
            )

            return global_similarity

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
