from typing import Dict

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import procrustes
from scipy.special import kl_div


def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calculate_curve_length_3d(points):
    """
    Calculates the length of a 3D curve defined by a list of 3D points.

    Args:
        points (list of tuples/lists): A list of 3D points (x, y, z).

    Returns:
        float: The total length of the curve.
    """
    # Convert points to a numpy array for easier manipulation
    trajectory = np.array(points)

    # Calculate the pairwise Euclidean distances between consecutive points
    distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)

    # Sum the distances to get the total length of the curve
    curve_length = np.sum(distances)

    return curve_length


def calculate_frechet_distance_3d(P, Q):
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


def calculate_frechet_similarity(
    trajectory_a: np.ndarray, trajectory_b: np.ndarray
) -> float:
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

    # Compute Fréchet distance between the two trajectories
    frechet_dist = calculate_frechet_distance_3d(trajectory_a, trajectory_b)

    # Compute the normalized similarity score
    length_product = np.sqrt(
        calculate_curve_length_3d(trajectory_a)
        * calculate_curve_length_3d(trajectory_b)
    )

    # Calculate similarity using the given formula
    similarity = 1 - (frechet_dist / length_product)

    # Ensure non-negative result (Fréchet distance can cause similarity < 0)
    return max(0, similarity)


def interp1d_array(array: np.ndarray, length: int):
    # Determine the new length (based on the longer array)
    # Resample array1 along the first axis (N-axis)
    x = np.linspace(0, 1, len(array))
    x_new = np.linspace(0, 1, length)
    array_resampled = interp1d(x, array, axis=0)(x_new)

    return array_resampled


def calculate_disparity_based_similarity(
    trajectory_a: np.ndarray, trajectory_b: np.ndarray, length: int = 1000
) -> float:
    """
    Compute the disparity based similarity between two trajectories.

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
        Calculated as 1 / (1 + disparity).
    """
    resample_a = interp1d_array(trajectory_a, length=length)
    resample_b = interp1d_array(trajectory_b, length=length)
    _, _, disparity = procrustes(resample_a, resample_b)
    return 1.0 / (1.0 + disparity)


def calculate_kl_divergence_similarity(
    trajectory_a: np.ndarray, trajectory_b: np.ndarray
) -> float:
    """
    Calculate a similarity score based on KL divergence between two probability distributions.

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
    """
    similarity_score = 1 / (1 + kl_div(trajectory_a, trajectory_b))

    return similarity_score


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


def calculate_trajectory_statistics(
    xyz_array: np.array,
    dt: np.array = None,
    sample_rate: float = None,
) -> Dict:
    """
    Compute the Fréchet similarity between two trajectories.

    Parameters
    ----------
    xyz_array : np.ndarray
        A Nx3 array representing the trajectory.
    dt : np.ndarray
        A N array representing the time difference.
    sample_rate: float
        Sample rate of trajectory


    Returns
    -------
    dict
    """
    x, y, z = xyz_array[:, 0], xyz_array[:, 1], xyz_array[:, 2]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, z)

    # Compute the differences in position and time (dx, dy, dz, dt)
    dx, dy, dz = np.diff(x), np.diff(y), np.diff(z)

    # Compute velocity components and magnitude
    if dt is not None:
        vx, vy, vz = dx / dt, dy / dt, dz / dt
    elif sample_rate is not None:
        vx, vy, vz = dx * sample_rate, dy * sample_rate, dz * sample_rate

    v = np.sqrt(vx**2 + vy**2 + vz**2)

    x_statistics = compute_statistics(x)
    x_min, x_max, x_mean, x_std_dev = (
        x_statistics.get("min"),
        x_statistics.get("max"),
        x_statistics.get("mean"),
        x_statistics.get("std_dev"),
    )
    y_statistics = compute_statistics(y)
    y_min, y_max, y_mean, y_std_dev = (
        y_statistics.get("min"),
        y_statistics.get("max"),
        y_statistics.get("mean"),
        y_statistics.get("std_dev"),
    )
    z_statistics = compute_statistics(z)
    z_min, z_max, z_mean, z_std_dev = (
        x_statistics.get("min"),
        z_statistics.get("max"),
        z_statistics.get("mean"),
        z_statistics.get("std_dev"),
    )
    r_statistics = compute_statistics(r)
    r_min, r_max, r_mean, r_std_dev = (
        r_statistics.get("min"),
        r_statistics.get("max"),
        r_statistics.get("mean"),
        r_statistics.get("std_dev"),
    )
    theta_statistics = compute_statistics(theta)
    theta_min, theta_max, theta_mean, theta_std_dev = (
        theta_statistics.get("min"),
        theta_statistics.get("max"),
        theta_statistics.get("mean"),
        theta_statistics.get("std_dev"),
    )
    v_statistics = compute_statistics(v)
    v_min, v_max, v_mean, v_std_dev = (
        v_statistics.get("min"),
        v_statistics.get("max"),
        v_statistics.get("mean"),
        v_statistics.get("std_dev"),
    )

    # Curve Length
    curve_length = calculate_curve_length_3d(xyz_array)

    return {
        "curveLength": curve_length,
        "xMin": x_min,
        "xMax": x_max,
        "xMean": x_mean,
        "xStdDev": x_std_dev,
        "yMin": y_min,
        "yMax": y_max,
        "yMean": y_mean,
        "yStdDev": y_std_dev,
        "zMin": z_min,
        "zMax": z_max,
        "zMean": z_mean,
        "zStdDev": z_std_dev,
        "rMin": r_min,
        "rMax": r_max,
        "rMean": r_mean,
        "rStdDev": r_std_dev,
        "thetaMin": theta_min,
        "thetaMax": theta_max,
        "thetaMean": theta_mean,
        "thetaStdDev": theta_std_dev,
        "vMin": v_min,
        "vMax": v_max,
        "vMean": v_mean,
        "vStdDev": v_std_dev,
    }
