from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from crowndata_evaluation.services.utils import fetch_trajectory_json
from crowndata_evaluation.services.compare_metric import dual_state_similarity

compare_metric_router = APIRouter()


# Request model
class EvaluationCompareMetricRequest(BaseModel):
    data1: Optional[List[List[float]]] = Field(
        None,
        example=[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ],
    )
    dataName1: Optional[str] = Field(None, example="droid_00000000")
    data2: Optional[List[List[float]]] = Field(
        None,
        example=[
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [0.1, 1.1, 2.1, 3.1, 4.1, 5.1],
        ],
    )
    dataName2: Optional[str] = Field(None, example="droid_00000001")

    # Custom validator to ensure each inner list has exactly 6 elements
    @validator("data1")
    def check_data1_length(cls, v):
        if v is not None:
            for inner_list in v:
                if len(inner_list) != 6:
                    raise HTTPException(
                        status_code=400,
                        detail="Data1 validation error: each inner list must have exactly 6 elements.",
                    )
        return v

    @validator("data2")
    def check_data2_length(cls, v):
        if v is not None:
            for inner_list in v:
                if len(inner_list) != 6:
                    raise HTTPException(
                        status_code=400,
                        detail="Data2 validation error: each inner list must have exactly 6 elements.",
                    )
        return v


# Response model
class EvaluationCompareMetricResponse(BaseModel):
    similarityScore: Optional[float]
    cosineSimilarityScore: Optional[float]


# POST endpoint for evaluating metrics
@compare_metric_router.post(
    "",
    summary="Compare Metrics",
    description="Compare two data",
    response_model=EvaluationCompareMetricResponse,
)
async def compare_metric(request: EvaluationCompareMetricRequest):
    # Ensure both data1/dataName1 and data2/dataName2 conditions are met
    if (request.data1 is not None and request.dataName1 is not None) or (
        request.data1 is None and request.dataName1 is None
    ):
        raise HTTPException(
            status_code=400,
            detail="Provide either 'data1' or 'dataName1', not both or neither.",
        )

    if (request.data2 is not None and request.dataName2 is not None) or (
        request.data2 is None and request.dataName2 is None
    ):
        raise HTTPException(
            status_code=400,
            detail="Provide either 'data2' or 'dataName2', not both or neither.",
        )

    # Handle data1
    data1 = None
    if request.data1 is not None:
        data1 = request.data1
    elif request.dataName1 is not None:
        data1 = fetch_trajectory_json(data_name=request.dataName1)

    # Handle data2
    data2 = None
    if request.data2 is not None:
        data2 = request.data2
    elif request.dataName2 is not None:
        data2 = fetch_trajectory_json(data_name=request.dataName2)

    # Ensure data is available before processing
    if data1 is None or data2 is None:
        raise HTTPException(
            status_code=400, detail="Both data1 and data2 must be provided."
        )

    # Ensure data1 and data2 are correctly formatted and non-empty
    similarity_score, cosine_similarity_score = dual_state_similarity(
        traj_a=data1, traj_b=data2, n_clusters=5, random_state=42
    )

    return {
        "similarityScore": round(similarity_score, 4),
        "cosineSimilarityScore": round(cosine_similarity_score, 4),
    }
