from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np
from crowndata_evaluation.services.utils import (
    fetch_trajectory_json,
    fetch_trajectory_sample_rate,
)
from crowndata_evaluation.services.action_consistency.action_variance_calculator import (
    ActionVarianceCalculator,
)
from crowndata_evaluation.services.action_consistency.state_similarity_calculator import (
    StateSimilarityCalculator,
)
from crowndata_evaluation.routers.v1.evaluation.helper import PoseData, JointData

group_metric_router = APIRouter()


# Request model
class EvaluationGroupMetricRequest(BaseModel):
    dataNames: List[PoseData] = Field(
        None,
        example=[
            {"dataName": "droid_00000000", "joints": ["cartesian_position"]},
            {"dataName": "droid_00000001", "joints": ["cartesian_position"]},
            {"dataName": "droid_00000002", "joints": ["cartesian_position"]},
        ],
    )


# Response model
class EvaluationGroupMetric(BaseModel):
    averageActionConsistency: Optional[float]
    averageStateSimilarityScore: Optional[float]


class EvaluationGroupMetricResponse(BaseModel):
    evaluationMetric: List[EvaluationGroupMetric]


# POST endpoint for evaluating metrics
@group_metric_router.post(
    "",
    summary="Group Metric",
    description="Retrieve group metric",
    response_model=EvaluationGroupMetricResponse,
)
async def group_metric(request: EvaluationGroupMetricRequest):
    # Check if 'dataNames' are provided and there are more than 3 items
    if not request.dataNames or len(request.dataNames) < 3:
        raise HTTPException(
            status_code=400,
            detail="Provide more than 3 'dataNames'.",
        )

    joint_length = len(request.dataNames[0].joints)
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
        xyz_data = [joint_data.data[:, :3] for joint_data in data_list[i]]

        avc = ActionVarianceCalculator(r=0.01)
        avg_action_consistency = 0

        for data in xyz_data:
            avg_action_consistency += avc.calculate_action_variance(data)

        avg_action_consistency /= len(request.dataNames)

        ssc = StateSimilarityCalculator(r=0.01, epsilon=0.1)

        number_of_groups = 5

        # Split xyz_data into 5 groups (xyz_groups will be a list of lists)
        xyz_groups = [xyz_data[i::number_of_groups] for i in range(number_of_groups)]

        # Iterate over the 5 groups and process them
        group_similarities = []
        similarities = []
        for grp, group in enumerate(xyz_groups):
            group_sim = []
            for i, xyz_array in enumerate(group):
                # Combine all other groups except the current group
                remaining_data = []
                for other_grp in range(number_of_groups):
                    if other_grp != grp:
                        for item in xyz_groups[other_grp]:
                            remaining_data.append(item)
                ssc.get_clusters(remaining_data)
                similarity = ssc.compute_trajectory_similarity(xyz_array)
                group_sim.append(similarity)
                similarities.append(similarity)
            group_similarities.append(group_sim)

        average_similarity_score = np.nanmean(similarities)

        evaluation_metric.append(
            {
                "averageActionConsistency": round(avg_action_consistency, 4),
                "averageStateSimilarityScore": round(average_similarity_score, 4),
            }
        )

    return {"evaluationMetric": evaluation_metric}
