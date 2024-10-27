from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np
from crowndata_evaluation.services.utils import fetch_trajectory_json
from crowndata_evaluation.services.action_consistency.action_variance_calculator import (
    ActionVarianceCalculator,
)
from crowndata_evaluation.services.action_consistency.state_similarity_calculator import (
    StateSimilarityCalculator,
)

group_metric_router = APIRouter()


# Request model
class EvaluationGroupMetricRequest(BaseModel):
    dataNames: Optional[List[str]] = Field(
        None, example=["droid_00000000", "droid_00000001", "droid_00000002"]
    )


# Response model
class EvaluationGroupMetricResponse(BaseModel):
    averageActionConsistency: Optional[float]
    averageStateSimilarityScore: Optional[float]


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
    avc = ActionVarianceCalculator(r=0.01)
    avg_action_consistency = 0

    data = []
    xyz_data = []
    for data_name in request.dataNames:
        data_item = fetch_trajectory_json(data_name=data_name)
        data.append(data_item)
        xyz_data.append(data_item[:, :3])
        avg_action_consistency += avc.calculate_action_variance(data_item)

    avg_action_consistency /= len(request.dataNames)

    ssc = StateSimilarityCalculator(r=0.01, epsilon=0.1)

    number_of_groups = 5
    # Assuming xyz_data is a list, calculate group size
    group_size = len(xyz_data) // number_of_groups

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

    return {
        "averageActionConsistency": round(avg_action_consistency, 4),
        "averageStateSimilarityScore": round(average_similarity_score, 4),
    }
