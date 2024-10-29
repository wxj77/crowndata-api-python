from typing import List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator

from crowndata_evaluation.routers.v1.helper import JointData, PoseData
from crowndata_evaluation.services.action_consistency.state_similarity_calculator import (
    StateSimilarityCalculator,
)
from crowndata_evaluation.services.shape.geometry import (
    calculate_disparity_based_similarity,
    calculate_frechet_similarity,
)
from crowndata_evaluation.services.utils import (
    fetch_trajectory_json,
    fetch_trajectory_sample_rate,
)

compare_metric_router = APIRouter()


# Request model
class EvaluationCompareMetricRequest(BaseModel):
    dataList1: List[JointData] = Field(
        None,
        example=[
            {
                "data": [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                ],
                "sample_rate": 10.0,
                "name": "cartesian_position",
            }
        ],
    )
    dataName1: PoseData = Field(
        None,
        example={"dataName": "droid_00000000", "joints": ["cartesian_position"]},
    )

    dataList2: List[JointData] = Field(
        None,
        example=[
            {
                "data": [
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                    [0.1, 1.1, 2.1, 3.1, 4.1, 5.1],
                ],
                "sample_rate": 10.0,
                "name": "cartesian_position",
            },
        ],
    )
    dataName2: PoseData = Field(
        None,
        example={"dataName": "droid_00000001", "joints": ["cartesian_position"]},
    )

    # Custom validator to ensure each inner list has exactly 6 elements
    @validator("dataList1")
    def check_dataList1_length(cls, data_list_1):
        if data_list_1 is not None:
            for v in data_list_1:
                for inner_list in v.data:
                    if len(inner_list) != 6 and len(inner_list) != 7:
                        raise HTTPException(
                            status_code=400,
                            detail="Data validation error: each inner list must have exactly 6 or 7 elements.",
                        )
            return data_list_1

    @validator("dataList2")
    def check_dataList2_length(cls, data_list_2):
        if data_list_2 is not None:
            for v in data_list_2:
                for inner_list in v.data:
                    if len(inner_list) != 6 and len(inner_list) != 7:
                        raise HTTPException(
                            status_code=400,
                            detail="Data validation error: each inner list must have exactly 6 or 7 elements.",
                        )
            return data_list_2


# Response model
class EvaluationCompareMetric(BaseModel):
    names: Optional[Tuple[str, str]] = None
    stateSimilarityScore: Optional[float]
    frechetSimilarityScore: Optional[float]
    disparityBasedSimilarityScore: Optional[float]


class EvaluationCompareMetricResponse(BaseModel):
    evaluationMetric: List[EvaluationCompareMetric]


# POST endpoint for evaluating metrics
@compare_metric_router.post(
    "",
    summary="Compare Metrics",
    description="Compare two data",
    response_model=EvaluationCompareMetricResponse,
)
async def compare_metric(request: EvaluationCompareMetricRequest):
    # Ensure both dataList1/dataName1 and dataList1/dataName2 conditions are met
    if (request.dataList1 is not None and request.dataName1 is not None) or (
        request.dataList1 is None and request.dataName1 is None
    ):
        raise HTTPException(
            status_code=400,
            detail="Provide either 'dataList1' or 'dataName1', not both or neither.",
        )

    if (request.dataList2 is not None and request.dataName2 is not None) or (
        request.dataList2 is None and request.dataName2 is None
    ):
        raise HTTPException(
            status_code=400,
            detail="Provide either 'dataList2' or 'dataName2', not both or neither.",
        )

    # Handle data1
    data_list_1 = []
    if request.dataList1 is not None:
        data_list_1 = request.dataList1
        for v in data_list_1:
            if len(v.data[0]) == 7:
                dt = v.data[0][:, 6]
                sample_rate = 1.0 / dt
                v.sample_rate = sample_rate
            v.data = np.array(v.data)
    elif request.dataName1 is not None:
        for k in request.dataName1.joints:
            data = fetch_trajectory_json(data_name=request.dataName1.dataName, joint=k)
            sample_rate = fetch_trajectory_sample_rate(
                data_name=request.dataName1.dataName
            )
            joint_data = JointData(data=data, sample_rate=sample_rate, name=k)
            data_list_1.append(joint_data)

    # Handle data2
    data_list_2 = []
    if request.dataList2 is not None:
        data_list_2 = request.dataList2
        for v in data_list_2:
            if len(v.data[0]) == 7:
                dt = v.data[0][:, 6]
                sample_rate = 1.0 / dt
                v.sample_rate = sample_rate
            v.data = np.array(v.data)
    elif request.dataName2 is not None:
        for k in request.dataName2.joints:
            data = fetch_trajectory_json(data_name=request.dataName2.dataName, joint=k)
            sample_rate = fetch_trajectory_sample_rate(
                data_name=request.dataName2.dataName
            )
            joint_data = JointData(data=data, sample_rate=sample_rate, name=k)
            data_list_2.append(joint_data)

    # Ensure data is available before processing
    if data_list_1 is None or data_list_2 is None:
        raise HTTPException(
            status_code=400, detail="Both data1 and data2 must be provided."
        )

    if len(data_list_1) - len(data_list_2) != 0:
        raise HTTPException(
            status_code=400, detail="data1 and data2 must be have same length."
        )

    evaluation_metric = []
    for i in range(len(data_list_1)):
        data1 = data_list_1[i].data
        data2 = data_list_2[i].data

        xyz_array1 = np.array(data1)[:, :3]
        xyz_array2 = np.array(data2)[:, :3]
        ssc = StateSimilarityCalculator(r=0.01, epsilon=0.1)
        ssc.get_clusters([xyz_array2])
        similarity1 = ssc.compute_trajectory_similarity(xyz_array1)
        ssc.get_clusters([xyz_array1])
        similarity2 = ssc.compute_trajectory_similarity(xyz_array2)
        similarities = [similarity1, similarity2]

        # Frechet Similarity
        frechet_similarity_score = calculate_frechet_similarity(xyz_array1, xyz_array2)

        # Disparity Similarity
        disparity_based_similarity_score = calculate_disparity_based_similarity(
            xyz_array1, xyz_array2
        )

        evaluation_metric.append(
            {
                "names": (
                    data_list_1[i].name,
                    data_list_2[i].name,
                ),
                "stateSimilarityScore": round(np.nanmean(similarities), 4),
                "frechetSimilarityScore": round(frechet_similarity_score, 4),
                "disparityBasedSimilarityScore": round(
                    disparity_based_similarity_score, 4
                ),
            }
        )

    return {"evaluationMetric": evaluation_metric}
