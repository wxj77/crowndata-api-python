import numpy as np
from typing import Dict
from crowndata_evaluation.services.action_consistency.clustering import define_clusters

from typing import Union, List, Dict
import numpy as np
from .clustering import define_clusters

class ActionVarianceCalculator:
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    # TODO 1: data input format
    def calculate_action_variance(self, states: Union[np.ndarray, List[List[float]]], actions: Union[np.ndarray, List[List[float]]]) -> float:
        # Convert inputs to numpy arrays if they're not already
        # TODO 2: add math formula
        # TODO 3: add type hints
        
        states = np.array(states) if not isinstance(states, np.ndarray) else states
        actions = np.array(actions) if not isinstance(actions, np.ndarray) else actions

        if states.shape[0] != actions.shape[0]:
            raise ValueError("Number of states and actions must be the same")

        if states.shape[0] == 0:
            return 0.0

        clusters: Dict[int, np.ndarray] = define_clusters(states, self.epsilon)

        if not clusters:
            return 0.0

        total_variance = 0.0
        total_points = 0

        for cluster in clusters.values():
            cluster_actions = actions[np.isin(states, cluster).all(axis=1)]
            if len(cluster_actions) > 1:
                cluster_variance = np.var(cluster_actions, axis=0).sum()
                total_variance += cluster_variance * len(cluster_actions)
                total_points += len(cluster_actions)

        if total_points == 0:
            return 0.0

       # TODO 4: Implement the action variance calculation
        average_variance = total_variance / total_points
        return average_variance
        # return [np.var(actions)]
 
