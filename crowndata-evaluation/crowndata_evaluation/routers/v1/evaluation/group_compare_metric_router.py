from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

group_compare_metric_router = APIRouter()


# Request model
class EvaluationGroupCompareMetricRequest(BaseModel):
    dataNames1: Optional[List[str]] = Field(
        None, example=["droid_00000000", "droid_00000001", "droid_00000002"]
    )
    dataNames2: Optional[List[str]] = Field(
        None, example=["droid_00000003", "droid_00000004", "droid_00000005"]
    )


# Response model
class EvaluationGroupCompareMetricResponse(BaseModel):
    similarityScore: Optional[float]


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
    # TODO: what does group mean?
    return {
        "similarityScore": -1,
    }
