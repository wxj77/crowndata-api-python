import numpy as np
from clustering import define_clusters


def test_define_clusters_basic():
    """
    Test basic clustering with a small epsilon where obvious clusters form.
    """
    data = np.array([[1, 2], [1.1, 2.1], [5, 6], [5.1, 6.2]])
    epsilon = 0.5
    expected_clusters = {
        0: np.array([[1.0, 2.0], [1.1, 2.1]]),
        1: np.array([[5.0, 6.0], [5.1, 6.2]]),
    }

    clusters = define_clusters(data, epsilon)

    # Check if clusters match the expected result
    for key in expected_clusters:
        np.testing.assert_array_equal(clusters[key], expected_clusters[key])


def test_define_clusters_no_clusters():
    """
    Test case where epsilon is too small to form any clusters.
    """
    data = np.array([[1, 2], [3, 4], [5, 6]])
    epsilon = 0.1  # Too small to form clusters
    expected_clusters = {
        0: np.array([[1, 2]]),
        1: np.array([[3, 4]]),
        2: np.array([[5, 6]]),
    }

    clusters = define_clusters(data, epsilon)

    # Check that each point forms its own cluster
    for key in expected_clusters:
        np.testing.assert_array_equal(clusters[key], expected_clusters[key])


def test_define_clusters_large_epsilon():
    """
    Test case where epsilon is large enough to put all points in a single cluster.
    """
    data = np.array([[1, 2], [1.1, 2.1], [5, 6]])
    epsilon = 10  # Large enough to group all points together
    expected_clusters = {0: np.array([[1.0, 2.0], [1.1, 2.1], [5.0, 6.0]])}

    clusters = define_clusters(data, epsilon)

    # Check if all points are grouped in a single cluster
    np.testing.assert_array_equal(clusters[0], expected_clusters[0])


def test_define_clusters_edge_case_empty_input():
    """
    Test case where the input data is empty.
    """
    data = np.array([])
    epsilon = 0.5
    clusters = define_clusters(data, epsilon)

    # Should return an empty dictionary since there's no data
    assert len(clusters) == 0
