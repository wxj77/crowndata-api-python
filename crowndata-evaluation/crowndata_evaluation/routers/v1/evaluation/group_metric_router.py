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

    data = []
    for data_name in request.dataNames:
        data_item = fetch_trajectory_json(data_name=data_name)
        data.append(data_item)
        avg_action_consistency += avc.calculate_action_variance(data_item)

    avg_action_consistency /= len(request.dataNames)

    ssc = StateSimilarityCalculator(epsilon=0.01)
    similarities = [ssc.compute_similarity(data_item, data) for data_item in data]

    return {
        "averageActionConsistency": round(avg_action_consistency, 4),
        "averageSimilarityScore": round(np.mean(similarities), 4),
    }
