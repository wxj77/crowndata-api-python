from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from crowndata_evaluation.services.utils import fetch_trajectory_json

compare_single_to_group_router = APIRouter()


# Request model
class EvaluationGroupCompareMetricRequest(BaseModel):
    dataName: Optional[str] = Field(None, example="droid_00000000")
    dataNames: Optional[List[str]] = Field(
        None, example=["droid_00000003", "droid_00000004", "droid_00000005"]
    )


# Response model
class EvaluationGroupCompareMetricResponse(BaseModel):
    similarityScore: Optional[float]


# POST endpoint for evaluating metrics
@compare_single_to_group_router.post(
    "",
    summary="Compare Single to Group Metric",
    description="Compare Single to Group",
    response_model=EvaluationGroupCompareMetricResponse,
)
async def compare_single_to_group_metric(request: EvaluationGroupCompareMetricRequest):
    # Check if 'dataNames' are provided and there are more than 3 items
    if not request.dataNames or len(request.dataNames) < 3:
        raise HTTPException(
            status_code=400,
            detail="Provide more than 3 data names in 'dataNames'.",
        )

    data_name = request.dataName
    data_item = fetch_trajectory_json(data_name=data_name)

    data = []
    for data_name in request.dataNames:
        data_item = fetch_trajectory_json(data_name=data_name)
        data.append(data_item)

    return {"similarityScore": 0.24}
