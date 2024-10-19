from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from crowndata_evaluation.services.action_consistency.action_variance_calculator import (
    ActionVarianceCalculator,
)
from crowndata_evaluation.services.utils import read_trajectory_json

metric_router = APIRouter()


# Request model
class EvaluationMetricRequest(BaseModel):
    data: Optional[List[List[float]]] = Field(
        None,
        example=[
            [
                3.83574843e-01,
                7.34695271e-02,
                5.51359415e-01,
                -2.89342165e00,
                -1.98712066e-01,
                1.26990348e-01,
            ],
            [
                3.83757949e-01,
                7.34741762e-02,
                5.52553594e-01,
                -2.89334679e00,
                -2.01124683e-01,
                1.26930386e-01,
            ],
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
        data = read_trajectory_json(data_name=request.dataName)

    avc = ActionVarianceCalculator(epsilon=0.1)
    action_consistency = avc.calculate_action_variance(data)

    return {
        "actionConsistency": round(action_consistency, 4),
    }
