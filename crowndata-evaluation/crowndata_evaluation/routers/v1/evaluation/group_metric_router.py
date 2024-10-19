from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from crowndata_evaluation.services.action_consistency.action_variance_calculator import (
    ActionVarianceCalculator,
)
from crowndata_evaluation.services.utils import read_trajectory_json

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
    avc = ActionVarianceCalculator(epsilon=0.1)
    avg_action_consistency = 0
    for data_name in request.dataNames:
        data_item = read_trajectory_json(data_name)
        avg_action_consistency += avc.calculate_action_variance(data_item)
    avg_action_consistency /= len(request.dataNames)
    return {
        "averageActionConsistency": round(avg_action_consistency, 4),
    }
