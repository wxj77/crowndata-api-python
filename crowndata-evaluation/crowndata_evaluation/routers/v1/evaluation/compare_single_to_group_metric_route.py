from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

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

compare_single_to_group_router = APIRouter()


# Request model
class EvaluationGroupCompareMetricRequest(BaseModel):
    dataName: PoseData = Field(
        None,
        example={"dataName": "droid_00000000", "joints": ["cartesian_position"]},
    )
    dataNames: List[PoseData] = Field(
        None,
        example=[
            {"dataName": "droid_00000000", "joints": ["cartesian_position"]},
            {"dataName": "droid_00000001", "joints": ["cartesian_position"]},
            {"dataName": "droid_00000002", "joints": ["cartesian_position"]},
        ],
    )


# Response model
class EvaluationGroupCompareMetric(BaseModel):
    stateSimilarityScore: Optional[float]
    frechetSimilarityScore: Optional[float]
    disparityBasedSimilarityScore: Optional[float]


class EvaluationGroupCompareMetricResponse(BaseModel):
    evaluationMetric: List[EvaluationGroupCompareMetric]


# POST endpoint for evaluating metrics
@compare_single_to_group_router.post(
    "",
    summary="Compare Single to Group Metric",
    description="Compare Single to Group",
    response_model=EvaluationGroupCompareMetricResponse,
)
async def post(request: EvaluationGroupCompareMetricRequest):
    # Check if 'dataNames' are provided and there are more than 3 items
    if not request.dataNames or len(request.dataNames) < 3:
        raise HTTPException(
            status_code=400,
            detail="Provide more than 3 data names in 'dataNames'.",
        )

    joint_length = len(request.dataName.joints)
    data_list_single = []
    if request.dataName is not None:
        for k in request.dataName.joints:
            data = fetch_trajectory_json(data_name=request.dataName.dataName, joint=k)
            sample_rate = fetch_trajectory_sample_rate(
                data_name=request.dataName.dataName
            )
            joint_data = JointData(data=data, sample_rate=sample_rate, name=k)
            data_list_single.append(joint_data)

    # Handle data
    data_list = []
    for data_name in request.dataNames:
        if len(data_name.joints) - joint_length != 0:
            raise HTTPException(
                status_code=400,
                detail="Provide same number of joint in each data.",
            )
        for i, k in enumerate(data_name.joints):
            data = fetch_trajectory_json(data_name=data_name.dataName, joint=k)
            sample_rate = fetch_trajectory_sample_rate(data_name=data_name.dataName)
            joint_data = JointData(data=data, sample_rate=sample_rate, name=k)
            if len(data_list) <= i:
                data_list.append([])
            data_list[i].append(joint_data)

    evaluation_metric = []

    for i in range(len(data_list)):
        xyz_array = data_list_single[i].data[:, :3]
        xyz_data = [joint_data.data[:, :3] for joint_data in data_list[i]]

        ssc = StateSimilarityCalculator(r=0.01, epsilon=0.1)
        ssc.get_clusters(xyz_data)
        # Ensure data1 and data2 are correctly formatted and non-empty
        similarity = ssc.compute_trajectory_similarity(xyz_array)

        # Frechet Similarity
        frechet_similarity_scores = []
        for xyz_array_item in xyz_data:
            frechet_similarity_score = calculate_frechet_similarity(
                xyz_array, xyz_array_item
            )
            frechet_similarity_scores.append(frechet_similarity_score)

        # Disparity Similarity
        disparity_based_similarity_scores = []
        for xyz_array_item in xyz_data:
            disparity_based_similarity_score = calculate_disparity_based_similarity(
                xyz_array, xyz_array_item
            )
            disparity_based_similarity_scores.append(disparity_based_similarity_score)

        evaluation_metric.append(
            {
                "stateSimilarityScore": float(round(similarity, 4)),
                "frechetSimilarityScore": float(
                    round(np.nanmean(frechet_similarity_scores), 4)
                ),
                "disparityBasedSimilarityScore": float(
                    round(np.nanmean(disparity_based_similarity_scores), 4)
                ),
            }
        )

    return {"evaluationMetric": evaluation_metric}
