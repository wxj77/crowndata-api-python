from fastapi.testclient import TestClient
from crowndata_evaluation.main import app
import pytest

client = TestClient(app)


@pytest.mark.parametrize(
    "payload, expected_status, expected_response",
    [
        (
            {"dataName": "droid_00000000", "data": []},
            422,
            None,
        ),
        ({"dataName": "not_exist_data"}, 422, None),
        ({"dataName": 0}, 422, None),
        ({"data": 0}, 400, None),
        (
            {
                "dataName": {
                    "dataName": "droid_00000000",
                    "joints": ["cartesian_position"],
                }
            },
            200,
            None,
        ),
        (
            {
                "dataList": [
                    {
                        "data": [
                            [1, 2, 3, 4, 5, 6],
                            [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
                            [0, 1, 2, 3, 4, 5],
                            [0.1, 1.1, 2.1, 3.1, 4.1, 5.1],
                        ],
                        "name": "cartesian_position",
                        "sample_rate": 10,
                    }
                ]
            },
            200,
            None,
        ),
    ],
)
def test_metric_router(payload, expected_status, expected_response):
    response = client.post("/v1/evaluation/metrics", json=payload)
    assert response.status_code == expected_status
    if expected_response:
        assert response.json() == expected_response


@pytest.mark.parametrize(
    "payload, expected_status, expected_response",
    [
        (
            {"dataName1": "droid_00000000", "data1": []},
            422,
            None,
        ),
        (
            {"dataName1": "not_exist_data", "dataName2": "droid_00000001"},
            422,
            None,
        ),
        (
            {
                "dataName1": {
                    "dataName": "droid_00000000",
                    "joints": ["cartesian_position"],
                },
                "dataName2": {
                    "dataName": "droid_00000001",
                    "joints": ["cartesian_position"],
                },
            },
            200,
            None,
        ),
        (
            {
                "dataList1": [
                    {
                        "data": [
                            [0, 0, 0, 0, 0, 0],
                            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                            [0, 0, 0, 0, 0, 0],
                            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                        ],
                        "name": "cartesian_position",
                        "sample_rate": 10,
                    }
                ],
                "dataList2": [
                    {
                        "data": [
                            [1, 2, 3, 4, 5, 6],
                            [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
                            [0, 1, 2, 3, 4, 5],
                            [0.1, 1.1, 2.1, 3.1, 4.1, 5.1],
                        ],
                        "name": "cartesian_position",
                        "sample_rate": 10,
                    }
                ],
            },
            200,
            None,
        ),
    ],
)
def test_compare_metric_router(payload, expected_status, expected_response):
    response = client.post("/v1/evaluation/compare-metrics", json=payload)
    assert response.status_code == expected_status
    if expected_response:
        assert response.json() == expected_response


@pytest.mark.parametrize(
    "payload, expected_status, expected_response",
    [
        (
            {
                "dataNames": "droid_00000000",
            },
            422,
            None,
        ),
        (
            {
                "dataNames": ["droid_00000000"],
            },
            422,
            None,
        ),
        (
            {
                "dataNames": [
                    {"dataName": "droid_00000000", "joints": ["cartesian_position"]},
                    {"dataName": "droid_00000001", "joints": ["cartesian_position"]},
                    {"dataName": "droid_00000002", "joints": ["cartesian_position"]},
                ]
            },
            200,
            None,
        ),
    ],
)
def test_group_metric_router(payload, expected_status, expected_response):
    response = client.post("/v1/evaluation/group-metrics", json=payload)
    assert response.status_code == expected_status
    if expected_response:
        assert response.json() == expected_response


@pytest.mark.parametrize(
    "payload, expected_status, expected_response",
    [
        (
            {
                "dataNames1": [
                    "droid_00000000",
                    "droid_00000001",
                ],
                "dataNames2": ["droid_00000003", "droid_00000004", "droid_00000005"],
            },
            422,
            None,
        ),
        (
            {
                "dataNames1": ["droid_00000000", "droid_00000001", "droid_00000002"],
            },
            422,
            None,
        ),
        (
            {
                "dataNames1": [
                    {"dataName": "droid_00000000", "joints": ["cartesian_position"]},
                    {"dataName": "droid_00000001", "joints": ["cartesian_position"]},
                    {"dataName": "droid_00000002", "joints": ["cartesian_position"]},
                ],
                "dataNames2": [
                    {"dataName": "droid_00000003", "joints": ["cartesian_position"]},
                    {"dataName": "droid_00000004", "joints": ["cartesian_position"]},
                    {"dataName": "droid_00000005", "joints": ["cartesian_position"]},
                ],
            },
            200,
            None,
        ),
    ],
)
def test_group_compare_metric_router(payload, expected_status, expected_response):
    response = client.post("/v1/evaluation/group-compare-metrics", json=payload)
    assert response.status_code == expected_status
    if expected_response:
        assert response.json() == expected_response


@pytest.mark.parametrize(
    "payload, expected_status, expected_response",
    [
        (
            {
                "dataNames": ["droid_00000003", "droid_00000004", "droid_00000005"],
            },
            422,
            None,
        ),
        (
            {
                "dataName": "droid_00000000",
            },
            422,
            None,
        ),
        (
            {
                "dataName": {
                    "dataName": "droid_00000003",
                    "joints": ["cartesian_position"],
                },
                "dataNames": [
                    {"dataName": "droid_00000000", "joints": ["cartesian_position"]},
                    {"dataName": "droid_00000001", "joints": ["cartesian_position"]},
                    {"dataName": "droid_00000002", "joints": ["cartesian_position"]},
                ],
            },
            200,
            None,
        ),
    ],
)
def test_compare_single_to_group_router(payload, expected_status, expected_response):
    response = client.post(
        "/v1/evaluation/compare-single-to-group-metrics", json=payload
    )
    assert response.status_code == expected_status
    if expected_response:
        assert response.json() == expected_response
