import pytest
import numpy as np
from crowndata_evaluation.services.action_consistency.clustering import define_clusters


@pytest.mark.parametrize(
    "data, epsilon, expected_num_clusters",
    [
        # Edge case: large epsilon = 1 big cluster
        [
            np.random.rand(100, 6),
            10000,
            1,
        ],
        # Edge case: small epsilon = 100 small clusters
        [
            np.random.rand(100, 6),
            0,
            100,
        ],
        [
            np.array(
                [
                    np.ones(6),
                    np.ones(6) + 0.1,
                ]
            ),
            0.25,
            1,
        ],
        [
            np.array(
                [
                    np.ones(6),
                    np.ones(6) + 0.1,
                ]
            ),
            0.24,
            2,
        ],
    ],
)
def test_define_clusters(data, epsilon, expected_num_clusters):
    """Test the define_clusters function."""
    clusters = define_clusters(data, epsilon)
    assert len(clusters) == expected_num_clusters
