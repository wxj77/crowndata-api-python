import numpy as np
from typing import Dict
from .clustering import define_clusters


class ActionVarianceCalculator:
    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon

    def calculate_action_variance(
        self, states: np.ndarray, actions: np.ndarray
    ) -> Dict[int, float]:
        if len(states) == 0 or len(actions) == 0:
            return {}

        clusters: Dict[int, np.ndarray] = define_clusters(states, self.epsilon)

        variances = {}
        for cluster_id, cluster_points in clusters.items():
            cluster_indices = np.array(
                [
                    np.where((states == point).all(axis=1))[0][0]
                    for point in cluster_points
                ]
            )
            cluster_actions = actions[cluster_indices]
            if len(cluster_actions) > 1:  # Need at least 2 points to calculate variance
                variance = np.var(cluster_actions, axis=0)
                variances[cluster_id] = float(np.mean(variance))
            else:
                variances[cluster_id] = 0.0

        return variances
