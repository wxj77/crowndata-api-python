import numpy as np
import pytest
from crowndata_evaluation.services.action_consistency.action_variance_calculator import (
    ActionVarianceCalculator,
)


@pytest.mark.parametrize(
    "traj_a, traj_b, variance",
    [
        [
            [
                [
                    -2.89342165,
                    -0.198712066,
                    0.126990348,
                    -2.89334679,
                    -0.201124683,
                    0.126930386,
                ]
            ],
            [
                [
                    -2.89342165,
                    -0.198712066,
                    0.126990348,
                    -2.89334679,
                    -0.201124683,
                    0.126930386,
                ]
            ],
            0,
        ]
    ],
)
def test_same_trajectory_action_variance(traj_a, traj_b, variance):
    calculator = ActionVarianceCalculator(epsilon=0.1)
    combined_traj = np.array(traj_a + traj_b)
    assert calculator.calculate_action_variance(combined_traj) == variance


# TODO 1: TEST if empty list is handled correctly
