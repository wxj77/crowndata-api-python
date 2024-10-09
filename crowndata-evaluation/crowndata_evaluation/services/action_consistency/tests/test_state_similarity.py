import pytest
import os
import numpy as np
from utils import read_json_cartesian_pose
from state_similarity_calculator import StateSimilarityCalculator


# TODO: any better way to load data from two JSON files only once?
@pytest.fixture(scope="module")
def load_data():
    """
    Fixture to load and return the data from two JSON files only once.
    """
    file_path1 = "crowndata-evaluation/example_data/data/droid_00000000/trajectories/cartesian_position__trajectory.json"
    file_path2 = "crowndata-evaluation/example_data/data/droid_00000001/trajectories/cartesian_position__trajectory.json"

    # Load data from both JSON files
    data1 = read_json_cartesian_pose(file_path1)
    data2 = read_json_cartesian_pose(file_path2)

    return data1, data2

def test_file_loading_and_validation(load_data):
    """
    Test loading and validation of data from the fixture.
    """
    data1, data2 = load_data

    # Verify that the data is correctly loaded as NumPy arrays with the correct dimensions
    assert isinstance(data1, np.ndarray), "Expected data1 to be a NumPy array"
    assert isinstance(data2, np.ndarray), "Expected data2 to be a NumPy array"
    assert (
        data1.shape[1] == 6
    ), "Each entry in data1 should contain 6 values (x, y, z, roll, pitch, yaw)"
    assert (
        data2.shape[1] == 6
    ), "Each entry in data2 should contain 6 values (x, y, z, roll, pitch, yaw)"


@pytest.mark.parametrize("epsilon", [0.5, 1.0])
def test_state_similarity_single_trajectory(epsilon, load_data):
    """
    Test state similarity for a single trajectory using the StateSimilarityCalculator.
    """
    data1, _ = load_data

    # Initialize StateSimilarityCalculator with a given epsilon
    calc = StateSimilarityCalculator(epsilon)

    # Compute similarity for the dataset
    similarity = calc.compute_similarity(data1)

    # Assert that the similarity score is within the valid range [0, 1]
    assert (
        0 <= similarity <= 1
    ), f"Expected similarity to be between 0 and 1, but got {similarity}"


@pytest.mark.parametrize("epsilon", [0.5])
def test_combined_trajectory_similarity(epsilon, load_data):
    """
    Test similarity between two combined trajectories.
    """
    data1, data2 = load_data

    # Initialize StateSimilarityCalculator with a given epsilon
    calc = StateSimilarityCalculator(epsilon)

    # Compute similarity between the combined datasets
    combined_data = np.vstack((data1, data2))
    combined_similarity = calc.compute_similarity(combined_data)

    # Assert that the combined similarity score is within the valid range [0, 1]
    assert (
        0 <= combined_similarity <= 1
    ), f"Expected combined similarity to be between 0 and 1, but got {combined_similarity}"


def test_empty_data_similarity():
    """
    Test state similarity with an empty dataset.
    """
    # Create an empty dataset
    empty_data = np.array([]).reshape(0, 6)

    # Initialize StateSimilarityCalculator with a given epsilon
    calc = StateSimilarityCalculator(epsilon=0.5)

    # Compute similarity for the empty dataset
    similarity = calc.compute_similarity(empty_data)

    # Assert that the similarity score is 0 for empty data
    assert similarity == 0, "Expected similarity for empty data to be 0"
