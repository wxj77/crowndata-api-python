import numpy as np
import pytest
from crowndata_evaluation.services.action_consistency.action_variance_calculator import (
    ActionVarianceCalculator,
)


@pytest.fixture
def action_variance_calculator():
    return ActionVarianceCalculator(epsilon=1.0)


def test_calculate_action_variance():
    # Create an instance of ActionVarianceCalculator
    calculator = ActionVarianceCalculator(epsilon=1.0)

    # Create sample states (xyzrpy)
    states = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0.5, 0.5, 0.5, 0.1, 0.1, 0.1],
            [0.8, 0.8, 0.8, 0.2, 0.2, 0.2],
            [5, 5, 5, 1, 1, 1],
            [5.5, 5.5, 5.5, 1.1, 1.1, 1.1],
        ]
    )

    # Create sample actions (also 6-dimensional)
    actions = np.array(
        [
            [0, 0.1, 0.2, 0.01, 0.02, 0.03],
            [0.1, 0.2, 0.3, 0.02, 0.03, 0.04],
            [0.2, 0.3, 0.4, 0.03, 0.04, 0.05],
            [2, 2.1, 2.2, 0.5, 0.6, 0.7],
            [2.1, 2.2, 2.3, 0.6, 0.7, 0.8],
        ]
    )

    # Calculate action variance
    result = calculator.calculate_action_variance(states, actions)

    # Expected results:
    # We expect two clusters: one for the first three points, one for the last two
    assert len(result) == 2

    # Check the variances
    variances = list(result.values())

    # The first cluster should have a small variance
    assert 0 < variances[0] < 0.1

    # The second cluster should have a larger variance
    assert 0 < variances[1] < 0.1

    # The second cluster's variance should be smaller than the first
    assert variances[1] < variances[0]


def test_calculate_action_variance_single_point():
    calculator = ActionVarianceCalculator(epsilon=1.0)

    # Single state and action
    states = np.array([[1, 2, 3, 0.1, 0.2, 0.3]])
    actions = np.array([[0.5, 0.6, 0.7, 0.01, 0.02, 0.03]])

    result = calculator.calculate_action_variance(states, actions)

    assert len(result) == 1
    assert list(result.values())[0] == 0.0


def test_calculate_action_variance_no_data():
    calculator = ActionVarianceCalculator(epsilon=1.0)

    # Empty arrays
    states = np.array([])
    actions = np.array([])

    result = calculator.calculate_action_variance(states, actions)

    assert len(result) == 0
