from fastapi.testclient import TestClient
from crowndata_evaluation.main import app
import pytest

client = TestClient(app)


@pytest.mark.parametrize(
    "payload, expected_status, expected_response",
    [
        (
            {"dataName": "droid_00000000", "data": []},
            400,
            None,
        ),
        ({"dataName": "not_exist_data"}, 400, {"detail": "Data not exist"}),
        ({"dataName": 0}, 422, None),
        ({"data": 0}, 422, None),
        ({"dataName": "droid_00000000"}, 200, None),
        (
            {
                "data": [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
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
            400,
            None,
        ),
        (
            {"dataName1": "not_exist_data", "dataName2": "droid_00000001"},
            400,
            {"detail": "Data not exist"},
        ),
        ({"dataName1": "droid_00000000", "dataName2": "droid_00000001"}, 200, None),
        (
            {
                "data1": [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                ],
                "data2": [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
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
            400,
            None,
        ),
        (
            {"dataNames": ["droid_00000000", "droid_00000001", "droid_00000002"]},
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
            400,
            None,
        ),
        (
            {
                "dataNames1": ["droid_00000000", "droid_00000001", "droid_00000002"],
            },
            400,
            None,
        ),
        (
            {
                "dataNames1": ["droid_00000000", "droid_00000001", "droid_00000002"],
                "dataNames2": ["droid_00000003", "droid_00000004", "droid_00000005"],
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
            400,
            None,
        ),
        (
            {
                "dataName": "droid_00000000",
            },
            400,
            None,
        ),
        (
            {
                "dataName": "droid_00000000",
                "dataNames": ["droid_00000003", "droid_00000004", "droid_00000005"],
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
