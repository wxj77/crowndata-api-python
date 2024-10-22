import numpy as np
from shapely import LineString, frechet_distance
from typing import Dict


def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def frechet_distance_3d(P, Q):
    """
    Compute the discrete Fréchet distance between two 3D curves P and Q.
    P and Q are lists of 3D points.
    """

    # Number of points in each curve
    n = len(P)
    m = len(Q)

    # Create an (n x m) array initialized to -1 for storing subproblem solutions
    ca = np.ones((n, m)) * -1

    def recursive_frechet(i, j):
        """
        Recursive helper function with memoization to compute the discrete Fréchet distance.
        i, j: indices in curves P and Q
        """
        if ca[i, j] > -1:
            return ca[i, j]
        elif i == 0 and j == 0:
            ca[i, j] = euclidean_distance(P[0], Q[0])
        elif i > 0 and j == 0:
            ca[i, j] = max(recursive_frechet(i - 1, 0), euclidean_distance(P[i], Q[0]))
        elif i == 0 and j > 0:
            ca[i, j] = max(recursive_frechet(0, j - 1), euclidean_distance(P[0], Q[j]))
        elif i > 0 and j > 0:
            ca[i, j] = max(
                min(
                    recursive_frechet(i - 1, j),
                    recursive_frechet(i - 1, j - 1),
                    recursive_frechet(i, j - 1),
                ),
                euclidean_distance(P[i], Q[j]),
            )
        else:
            ca[i, j] = float("inf")
        return ca[i, j]

    # Call the recursive function for the full length of both curves
    return recursive_frechet(n - 1, m - 1)


def frechet_similarity(trajectory_a: np.ndarray, trajectory_b: np.ndarray) -> float:
    """
    Compute the Fréchet similarity between two trajectories.

    Parameters
    ----------
    trajectory_a : np.ndarray
        A Nx3 array representing the first trajectory.
    trajectory_b : np.ndarray
        A Nx3 array representing the second trajectory.

    Returns
    -------
    float
        A value between 0 and 1 representing the similarity, where 1 indicates identical trajectories.
        Calculated as max(0, 1 - (Fréchet distance / sqrt(curve_length(a) * curve_length(b)))).
    """

    trajectory_a = trajectory_a - np.mean(trajectory_a, axis=0)
    trajectory_b = trajectory_b - np.mean(trajectory_b, axis=0)

    # Convert trajectories into LineString objects
    line_a = LineString(trajectory_a)
    line_b = LineString(trajectory_b)

    # Compute Fréchet distance between the two trajectories
    frechet_dist = frechet_distance_3d(trajectory_a, trajectory_b)

    # Compute the normalized similarity score
    length_product = np.sqrt(line_a.length * line_b.length)

    # Calculate similarity using the given formula
    similarity = 1 - (frechet_dist / length_product)

    # Ensure non-negative result (Fréchet distance can cause similarity < 0)
    return max(0, similarity)


def compute_statistics(arr: np.ndarray) -> Dict:
    """
    Compute basic statistics for a given numpy array.

    Parameters:
    arr (numpy array): Input array

    Returns:
    dict: Dictionary containing computed statistics
    """
    statistics = {
        "mean": np.nanmean(arr),
        "median": np.nanmedian(arr),
        "std_dev": np.nanstd(arr),
        "variance": np.nanvar(arr),
        "min": np.nanmin(arr),
        "max": np.nanmax(arr),
        "sum": np.nansum(arr),
    }

    return statistics
