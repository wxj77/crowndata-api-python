from typing import Tuple
import numpy as np
from sklearn.cluster import KMeans
from crowndata_evaluation.services.utils import cosine_similarity


def dual_state_similarity(
    traj_a: np.ndarray, traj_b: np.ndarray, n_clusters=5, random_state=42
) -> Tuple[float, float]:
    """
    Compute the similarity between two trajectories using KMeans clustering.

    Args:
        traj_a (np.ndarray): Trajectory A
        traj_b (np.ndarray): Trajectory B

    Returns:
        float: Similarity between the two trajectories
    """
    # Combine both trajectories into one dataset for clustering
    combined_data = np.vstack((traj_a, traj_b))

    # Perform KMeans clustering on the combined data with random seed 42
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(combined_data)

    # Get cluster labels for each trajectory
    labels_a = kmeans.predict(traj_a)
    labels_b = kmeans.predict(traj_b)

    # Compute normalized histograms (cluster distributions) for both trajectories
    hist_a = np.bincount(labels_a, minlength=n_clusters) / len(traj_a)
    hist_b = np.bincount(labels_b, minlength=n_clusters) / len(traj_b)

    # Compute similarity as the inverse of the norm between the two histograms
    similarity_score = 1 - np.linalg.norm(hist_a - hist_b)
    cosine_similarity_score = cosine_similarity(hist_a, hist_b)

    return similarity_score, cosine_similarity_score
