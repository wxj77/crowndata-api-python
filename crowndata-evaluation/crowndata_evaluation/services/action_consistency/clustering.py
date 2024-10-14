import numpy as np
from typing import Dict
import sklearn.cluster
from scipy.spatial import cKDTree


def sklearn_cluster_wrapper(
    data: np.ndarray, class_name: str, args: Dict
) -> Dict[int, np.ndarray]:
    """Use sklearn.cluster to cluster data.

    Parameters
    ----------
    data : np.ndarray
        Data to cluster. (number of samples, state_dimension)
    class_name : str
        Name of the Clustering Algorithm class in sklearn.cluster to use.
    args : Dict
        Arguments to pass to the class or function.

    Returns
    -------
    Dict[int, np.ndarray]
        A dictionary where each key is a cluster index, and the value is the
        states or actions in that cluster.
    """
    # check if given name is in sklearn.cluster
    if not hasattr(sklearn.cluster, class_name):
        raise ValueError(f"{class_name} is not in sklearn.cluster")
    # check if class name has caps, otherwise it's a method
    if class_name.islower():
        raise ValueError(
            f"{class_name} does not contain any uppercase letters. Use the class version instead of the method."
        )

    clusters = {}
    clustering_alg = getattr(sklearn.cluster, class_name)(**args)
    labels = clustering_alg.fit_predict(data)

    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)

    for key in clusters:
        clusters[key] = data[clusters[key]]

    return clusters


# # TODO: add a name to the function
# def define_clusters(data: np.ndarray, epsilon: float) -> Dict[int, np.ndarray]:
#     r"""
#     Define clusters for a set of states or actions based on epsilon distance.

#     For each state/action ``x_i``, we define a cluster ``C(x_i)`` as:

#     .. math::
#         C(x_i) = \{ x_j | \|x_i - x_j\| \leq \epsilon \}

#     where :math:`\|x_i - x_j\|` is the Euclidean distance between state/action
#     ``x_i`` and ``x_j``.

#     Parameters
#     ----------
#     data : np.ndarray
#         State or action data to cluster. (x, y, z, rotation_x, rotation_y, rotation_z)
#     epsilon : float
#         The coverage distance for clustering.

#     Returns
#     -------
#     Dict[int, np.ndarray]
#         A dictionary where each key is a cluster index, and the value is the
#         states or actions in that cluster.
#     """

#     clusters = {}
#     cluster_idx = 0
#     visited = np.zeros(len(data), dtype=bool)

#     for i in range(len(data)):
#         if not visited[i]:
#             cluster = [i]
#             for j in range(i + 1, len(data)):
#                 if np.linalg.norm(data[i] - data[j]) <= epsilon:
#                     cluster.append(j)
#                     visited[j] = True
#             clusters[cluster_idx] = data[cluster]
#             cluster_idx += 1

#     return clusters


# Efficent NN search for clustering
def define_clusters(data: np.ndarray, epsilon: float) -> Dict[int, np.ndarray]:
    """Cluster data points based on epsilon distance using scipy's cKDTree.

    This function implements a density-based clustering algorithm similar to
    DBSCAN, using cKDTree for efficient nearest neighbor searches. It groups
    data points into clusters where points within each cluster are at most
    'epsilon' distance apart.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        The data to cluster.

    epsilon : float
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.

    Returns
    -------
    clusters : dict of {int : array-like}
        A dictionary where each key is a cluster index, and the value is an
        array of shape (n_samples_in_cluster, n_features) containing the points
        in that cluster, sorted for consistency.

    Notes
    -----
    The clustering process can be described mathematically as follows:

    1. For each point p in the dataset:
       N_ε(p) = {q ∈ D | dist(p,q) ≤ ε}
       where N_ε(p) is the ε-neighborhood of p, and D is the dataset.

    2. A cluster C is formed by connecting points that are density-reachable:
       p ∈ C, q ∈ C if ∃ p_1, ..., p_n ∈ D : p_1 = p, p_n = q and
       p_{i+1} ∈ N_ε(p_i) for i = 1, ..., n-1

    The implementation uses scipy's cKDTree for efficient nearest neighbor
    searches, which reduces the time complexity from O(n^2) to O(n log n)
    for the neighbor-finding step.

    """
    # ... (rest of the function implementation remains the same)
    if len(data) == 0:
        return {}

    tree = cKDTree(data)
    neighbors = tree.query_ball_tree(tree, r=epsilon)

    clusters: Dict[int, np.ndarray] = {}
    visited = np.zeros(len(data), dtype=bool)
    cluster_idx = 0

    for i in range(len(data)):
        if not visited[i]:
            cluster_points = []
            stack = [i]
            visited[i] = True

            while stack:
                current = stack.pop()
                cluster_points.append(current)

                for neighbor in neighbors[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)

            # Sort cluster points based on their indices
            cluster_points.sort()
            # Use sorted indices to get sorted data points
            clusters[cluster_idx] = data[cluster_points]
            cluster_idx += 1

    return clusters
