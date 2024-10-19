from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from crowndata_evaluation.services.utils import read_trajectory_json
from crowndata_evaluation.services.action_consistency.state_similarity_calculator import (
    StateSimilarityCalculator,
)

compare_metric_router = APIRouter()


# Request model
class EvaluationCompareMetricRequest(BaseModel):
    data1: Optional[List[List[float]]] = Field(
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
    dataName1: Optional[str] = Field(None, example="droid_00000000")
    data2: Optional[List[List[float]]] = Field(
        None,
        example=[
            [
                4.83218133e-01,
                7.00685307e-02,
                2.66119927e-01,
                3.01093650e00,
                -4.27649707e-01,
                2.29388833e-01,
            ],
            [
                4.81317997e-01,
                6.99858665e-02,
                2.67044962e-01,
                3.01118779e00,
                -4.25343573e-01,
                2.30850562e-01,
            ],
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
        data1 = read_trajectory_json(data_name=request.dataName1)

    # Handle data2
    data2 = None
    if request.data2 is not None:
        data2 = request.data2
    elif request.dataName2 is not None:
        data2 = read_trajectory_json(data_name=request.dataName2)

    # Ensure data is available before processing
    if data1 is None or data2 is None:
        raise HTTPException(
            status_code=400, detail="Both data1 and data2 must be provided."
        )

    ssc = StateSimilarityCalculator(epsilon=0.01)
    # Ensure data1 and data2 are correctly formatted and non-empty
    global_similarities = ssc.compute_similarity(
        trajectories={request.dataName1: data1, request.dataName2: data2}
    )

    return {
        "similarityScore": round(global_similarities, 4),
        "cosineSimilarityScore": round(global_similarities, 4),
    }
