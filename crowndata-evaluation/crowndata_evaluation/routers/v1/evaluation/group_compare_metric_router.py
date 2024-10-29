from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from crowndata_evaluation.routers.v1.helper import JointData, PoseData
from crowndata_evaluation.services.action_consistency.state_similarity_calculator import \
    StateSimilarityCalculator
from crowndata_evaluation.services.utils import (fetch_trajectory_json,
                                                 fetch_trajectory_sample_rate)

group_compare_metric_router = APIRouter()


# Request model
class EvaluationGroupCompareMetricRequest(BaseModel):
    dataNames1: List[PoseData] = Field(
        None,
        example=[
            {"dataName": "droid_00000000", "joints": ["cartesian_position"]},
            {"dataName": "droid_00000001", "joints": ["cartesian_position"]},
            {"dataName": "droid_00000002", "joints": ["cartesian_position"]},
        ],
    )
    dataNames2: List[PoseData] = Field(
        None,
        example=[
            {"dataName": "droid_00000003", "joints": ["cartesian_position"]},
            {"dataName": "droid_00000004", "joints": ["cartesian_position"]},
            {"dataName": "droid_00000005", "joints": ["cartesian_position"]},
        ],
    )


# Response model
class EvaluationGroupCompareMetric(BaseModel):
    stateSimilarityScore: Optional[float]


class EvaluationGroupCompareMetricResponse(BaseModel):
    evaluationMetric: List[EvaluationGroupCompareMetric]


# POST endpoint for evaluating metrics
@group_compare_metric_router.post(
    "",
    summary="Group Compare Metric",
    description="Retrieve group compare metric",
    response_model=EvaluationGroupCompareMetricResponse,
)
async def group_compare_metric(request: EvaluationGroupCompareMetricRequest):
    # Check if 'dataNames' are provided and there are more than 3 items
    if not request.dataNames1 or len(request.dataNames1) < 3:
        raise HTTPException(
            status_code=400,
            detail="Provide more than 3 data names in 'dataNames1'.",
        )

    if not request.dataNames2 or len(request.dataNames2) < 3:
        raise HTTPException(
            status_code=400,
            detail="Provide more than 3 data names in 'dataNames2'.",
        )

    joint_length = len(request.dataNames1[0].joints)
    # Handle data1
    data_list_1 = []
    for data_name in request.dataNames1:
        if len(data_name.joints) - joint_length != 0:
            raise HTTPException(
                status_code=400,
                detail="Provide same number of joint in each data.",
            )
        for i, k in enumerate(data_name.joints):
            data = fetch_trajectory_json(data_name=data_name.dataName, joint=k)
            sample_rate = fetch_trajectory_sample_rate(data_name=data_name.dataName)
            joint_data = JointData(data=data, sample_rate=sample_rate, name=k)
            if len(data_list_1) <= i:
                data_list_1.append([])
            data_list_1[i].append(joint_data)

    # Handle data2
    data_list_2 = []
    for data_name in request.dataNames2:
        if len(data_name.joints) - joint_length != 0:
            raise HTTPException(
                status_code=400,
                detail="Provide same number of joint in each data.",
            )
        for i, k in enumerate(data_name.joints):
            data = fetch_trajectory_json(data_name=data_name.dataName, joint=k)
            sample_rate = fetch_trajectory_sample_rate(data_name=data_name.dataName)
            joint_data = JointData(data=data, sample_rate=sample_rate, name=k)
            if len(data_list_2) <= i:
                data_list_2.append([])
            data_list_2[i].append(joint_data)

    # Ensure data is available before processing
    if data_list_1 is None or data_list_2 is None:
        raise HTTPException(
            status_code=400, detail="Both data1 and data2 must be provided."
        )

    evaluation_metric = []
    for i in range(len(data_list_1)):
        xyz_data1 = [joint_data.data[:, :3] for joint_data in data_list_1[i]]
        xyz_data2 = [joint_data.data[:, :3] for joint_data in data_list_2[i]]

        ssc = StateSimilarityCalculator(r=0.01, epsilon=0.1)
        ssc.get_clusters(xyz_data2)
        similarities1 = [
            ssc.compute_trajectory_similarity(xyz_array) for xyz_array in xyz_data1
        ]
        ssc.get_clusters(xyz_data1)
        similarities2 = [
            ssc.compute_trajectory_similarity(xyz_array) for xyz_array in xyz_data2
        ]

        evaluation_metric.append(
            {
                "stateSimilarityScore": float(
                    round(
                        np.nanmean(
                            [np.nanmean(similarities1), np.nanmean(similarities2)]
                        ),
                        4,
                    )
                ),
            }
        )

    return {"evaluationMetric": evaluation_metric}
