from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from crowndata_evaluation.services.utils import fetch_trajectory_json
from crowndata_evaluation.services.action_consistency.action_variance_calculator import (
    ActionVarianceCalculator,
)

metric_router = APIRouter()


# Request model
class EvaluationMetricRequest(BaseModel):
    data: Optional[List[List[float]]] = Field(
        None,
        example=[
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [0.1, 1.1, 2.1, 3.1, 4.1, 5.1],
        ],
    )
    dataName: Optional[str] = Field(None, example="droid_00000000")

    # Custom validator to ensure each inner list has exactly 6 elements
    @validator("data")
    def check_data_length(cls, v):
        if v is not None:
            for inner_list in v:
                if len(inner_list) != 6:
                    raise HTTPException(
                        status_code=400,
                        detail="Data validation error: each inner list must have exactly 6 elements.",
                    )
        return v


# Response model
class EvaluationMetricResponse(BaseModel):
    actionConsistency: Optional[float]


# POST endpoint for evaluating metrics
@metric_router.post(
    "",
    summary="Metric",
    description="Retrieve metric",
    response_model=EvaluationMetricResponse,
)
async def metric(request: EvaluationMetricRequest):
    # Check if both 'data' and 'dataName' are provided or neither is provided
    if (request.data is not None and request.dataName is not None) or (
        request.data is None and request.dataName is None
    ):
        # Raise an HTTPException with a 400 status code for bad request
        raise HTTPException(
            status_code=400,
            detail="Provide either 'data' or 'dataName', not both or neither.",
        )

    data = None
    if request.data is not None:
        data = request.data
    elif request.dataName is not None:
        data = fetch_trajectory_json(data_name=request.dataName)

    avc = ActionVarianceCalculator(epsilon=0.1)
    action_consistency = avc.calculate_action_variance(data)

    return {
        "actionConsistency": round(action_consistency, 4),
    }
