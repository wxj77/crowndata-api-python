import numpy as np
from typing import Union, Dict
from crowndata_evaluation.services.action_consistency.clustering import define_clusters


class ActionVarianceCalculator:
    """
    A class to compute the action variance of a dataset.

    Parameters
    ----------
    epsilon : float
        The coverage distance for clustering.
    """

    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def calculate_action_variance(
        self, trajectories: Union[Dict[str, np.ndarray], np.ndarray]
    ) -> float:
        """
        Calculate the action variance based on the formula:
        ActionVariance(D) = (1 / |D|) * sum((a - mean(a_cluster))^2)
        where D is the dataset, a is an action, and a_cluster are the actions in the same cluster as a.

        Parameters
        ----------
        trajectories : Union[Dict[str, np.ndarray], np.ndarray]
            A dictionary where keys are demo names and values are trajectories, or a single trajectory.

        Returns
        -------
        float
            The action variance of the dataset
        """

        all_states = np.vstack([np.array(traj) for traj in trajectories])

        if isinstance(trajectories, dict):
            all_states = np.vstack([traj for traj in trajectories.values()])
        else:
            all_states = np.vstack([np.array(traj) for traj in trajectories])

        clusters = define_clusters(all_states, self.epsilon)
        print("Clusters:", clusters)

        if len(clusters) == len(all_states):
            print("Each point is its own cluster. Using overall variance.")
            overall_mean = np.mean(all_states, axis=0)
            overall_variance = np.sum(np.sum((all_states - overall_mean) ** 2, axis=1))
            return overall_variance / len(all_states)

        total_variance = 0
        total_actions = 0

        # TODO: Double check if this is correct
        for cluster_id, cluster_actions in clusters.items():
            print(f"Cluster {cluster_id}:\n", cluster_actions)
            cluster_mean = np.mean(cluster_actions, axis=0)
            cluster_variance = np.sum(
                np.sum((cluster_actions - cluster_mean) ** 2, axis=1)
            )

            total_variance += cluster_variance
            total_actions += len(cluster_actions)

        print(f"Total variance: {total_variance:.2e}")
        print(f"Total actions: {total_actions}")

        action_variance = total_variance / total_actions if total_actions > 0 else 0
        print(f"Action variance: {action_variance:.2e}")

        return action_variance
