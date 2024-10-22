import numpy as np
from shapely import LineString, frechet_distance
from typing import Dict


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
    # Convert trajectories into LineString objects
    line_a = LineString(trajectory_a)
    line_b = LineString(trajectory_b)

    # Compute Fréchet distance between the two trajectories
    frechet_dist = frechet_distance(line_a, line_b)

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
