from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from crowndata_evaluation.services.metric import get_action_consistency
from crowndata_evaluation.services.utils import read_json_file

group_metric_router = APIRouter()


# Request model
class EvaluationGroupMetricRequest(BaseModel):
    dataNames: Optional[List[str]] = Field(
        None, example=["droid_00000000", "droid_00000001", "droid_00000002"]
    )


# Response model
class EvaluationGroupMetricResponse(BaseModel):
    averageActionConsistency: Optional[float]


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

    data = []
    action_consistencies = []
    for data_name in request.dataNames:
        file_path = f"./example_data/data/{data_name}/trajectories/cartesian_position__trajectory.json"
        data_item = read_json_file(file_path)
        action_consistency = get_action_consistency(data=data_item)
        action_consistencies.append(action_consistency)

    # Calculate the average of action consistencies
    if action_consistencies:
        average_action_consistency = sum(action_consistencies) / len(
            action_consistencies
        )
    else:
        average_action_consistency = 0

    return {
        "averageActionConsistency": round(average_action_consistency, 4),
    }
