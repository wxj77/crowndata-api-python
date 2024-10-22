from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np
from crowndata_evaluation.services.utils import fetch_trajectory_json
from crowndata_evaluation.services.action_consistency.state_similarity_calculator import (
    StateSimilarityCalculator,
)
from crowndata_evaluation.services.shape.geometry import (
    calculate_frechet_similarity,
    calculate_disparity_similarity,
)

compare_single_to_group_router = APIRouter()


# Request model
class EvaluationGroupCompareMetricRequest(BaseModel):
    dataName: Optional[str] = Field(None, example="droid_00000000")
    dataNames: Optional[List[str]] = Field(
        None, example=["droid_00000003", "droid_00000004", "droid_00000005"]
    )


# Response model
class EvaluationGroupCompareMetricResponse(BaseModel):
    stateSimilarityScore: Optional[float]
    frechetSimilarityScore: Optional[float]
    disparitySimilarityScore: Optional[float]


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
    single_data = fetch_trajectory_json(data_name=data_name)
    xyz_array = single_data[:, :3]

    data = []
    xyz_data = []
    for data_name in request.dataNames:
        data_item = fetch_trajectory_json(data_name=data_name)
        data.append(data_item)
        xyz_data.append(data_item[:, :3])

    ssc = StateSimilarityCalculator(r=0.01, epsilon=0.1)
    ssc.get_clusters(xyz_data)
    # Ensure data1 and data2 are correctly formatted and non-empty
    similarity = ssc.compute_trajectory_similarity(xyz_array)

    # Frechet Similarity
    frechet_similarity_scores = []
    for xyz_array_item in xyz_data:
        frechet_similarity_score = calculate_frechet_similarity(
            xyz_array, xyz_array_item
        )
        frechet_similarity_scores.append(frechet_similarity_score)

    # Disparity Similarity
    disparity_similarity_scores = []
    for xyz_array_item in xyz_data:
        disparity_similarity_score = calculate_disparity_similarity(
            xyz_array, xyz_array_item
        )
        disparity_similarity_scores.append(disparity_similarity_score)

    return {
        "stateSimilarityScore": round(similarity, 4),
        "frechetSimilarityScore": round(np.nanmean(frechet_similarity_scores), 4),
        "disparitySimilarityScore": round(np.nanmean(disparity_similarity_scores), 4),
    }
