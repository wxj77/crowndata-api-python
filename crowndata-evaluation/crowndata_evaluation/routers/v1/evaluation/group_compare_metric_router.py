from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np
from crowndata_evaluation.services.utils import fetch_trajectory_json
from crowndata_evaluation.services.action_consistency.state_similarity_calculator import (
    StateSimilarityCalculator,
)

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

    data1 = []
    for data_name in request.dataNames1:
        data_item = fetch_trajectory_json(data_name=data_name)
        data1.append(data_item)

    data2 = []
    for data_name in request.dataNames2:
        data_item = fetch_trajectory_json(data_name=data_name)
        data2.append(data_item)

    ssc = StateSimilarityCalculator(epsilon=0.01)
    similarities1 = [ssc.compute_similarity(data_item, data2) for data_item in data1]
    similarities2 = [ssc.compute_similarity(data_item, data1) for data_item in data2]

    return {
        "similarityScore": round(
            np.mean([np.mean(similarities1), np.mean(similarities2)]), 4
        ),
    }
