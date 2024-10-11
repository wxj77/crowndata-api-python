import numpy as np
from typing import Dict


def define_clusters(data: np.ndarray, epsilon: float) -> Dict[int, np.ndarray]:
    r"""
    Define clusters for a set of states or actions based on epsilon distance.

    For each state/action ``x_i``, we define a cluster ``C(x_i)`` as:

    .. math::
        C(x_i) = \{ x_j | \|x_i - x_j\| \leq \epsilon \}

    where :math:`\|x_i - x_j\|` is the Euclidean distance between state/action
    ``x_i`` and ``x_j``.

    Parameters
    ----------
    data : np.ndarray
        State or action data to cluster. (x, y, z, rotation_x, rotation_y, rotation_z)
    epsilon : float
        The coverage distance for clustering.

    Returns
    -------
    Dict[int, np.ndarray]
        A dictionary where each key is a cluster index, and the value is the
        states or actions in that cluster.
    """
    # TODO 1: Can we refer the scipy dbscan function to implement this?
    # TODO 2: Does the translation and rotation need to be considered separately to define clusters?
    # should we count distance (x,y,z) and distance(rotation_x, rotation_y, rotation_z) separately?
    # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/cluster/_dbscan.py
    # paper: https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf

    clusters = {}
    cluster_idx = 0
    visited = np.zeros(len(data), dtype=bool)

    # TODO: manual writen clustering, replace to use sklearn.cluster.DBSCAN for better correctness

    for i in range(len(data)):
        if not visited[i]:
            cluster = [i]
            for j in range(i + 1, len(data)):
                if np.linalg.norm(data[i] - data[j]) <= epsilon:
                    cluster.append(j)
                    visited[j] = True
            clusters[cluster_idx] = data[cluster]
            cluster_idx += 1

    return clusters
