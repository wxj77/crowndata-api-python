from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from crowndata_evaluation.services.utils import (
    fetch_trajectory_json,
    fetch_trajectory_sample_rate,
    default_sample_rate,
)
from crowndata_evaluation.services.action_consistency.action_variance_calculator import (
    ActionVarianceCalculator,
)
from crowndata_evaluation.services.shape.geometry import (
    compute_statistics,
    calculate_trajectory_statistics,
    calculate_curve_length_3d,
)
import numpy as np

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
    curveLength: float
    xMin: float
    xMax: float
    xMean: float
    xStdDev: float
    yMin: float
    yMax: float
    yMean: float
    yStdDev: float
    zMin: float
    zMax: float
    zMean: float
    zStdDev: float
    rMin: float
    rMax: float
    rMean: float
    rStdDev: float
    thetaMin: float
    thetaMax: float
    thetaMean: float
    thetaStdDev: float
    vMin: float
    vMax: float
    vMean: float
    vStdDev: float


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
    dt = None
    sample_rate = default_sample_rate
    if request.data is not None:
        data = request.data
        if len(data[0]) == 7:
            dt = data[:, 6]
    elif request.dataName is not None:
        data = fetch_trajectory_json(data_name=request.dataName)
        sample_rate = fetch_trajectory_sample_rate(data_name=request.dataName)
    data = np.array(data)

    avc = ActionVarianceCalculator(r=0.1)
    action_consistency = avc.calculate_action_variance(data)

    xyz_array = data[:, :3]
    trajectory_statistics = calculate_trajectory_statistics(
        xyz_array=xyz_array,
        dt=dt,
        sample_rate=sample_rate,
    )

    return {
        "actionConsistency": round(action_consistency, 4),
        "curveLength": round(trajectory_statistics.get("curveLength"), 4),
        "xMin": round(trajectory_statistics.get("xMin"), 4),
        "xMax": round(trajectory_statistics.get("xMax"), 4),
        "xMean": round(trajectory_statistics.get("xMean"), 4),
        "xStdDev": round(trajectory_statistics.get("xStdDev"), 4),
        "yMin": round(trajectory_statistics.get("yMin"), 4),
        "yMax": round(trajectory_statistics.get("yMax"), 4),
        "yMean": round(trajectory_statistics.get("yMean"), 4),
        "yStdDev": round(trajectory_statistics.get("yStdDev"), 4),
        "zMin": round(trajectory_statistics.get("zMin"), 4),
        "zMax": round(trajectory_statistics.get("zMax"), 4),
        "zMean": round(trajectory_statistics.get("zMean"), 4),
        "zStdDev": round(trajectory_statistics.get("zStdDev"), 4),
        "rMin": round(trajectory_statistics.get("rMin"), 4),
        "rMax": round(trajectory_statistics.get("rMax"), 4),
        "rMean": round(trajectory_statistics.get("rMean"), 4),
        "rStdDev": round(trajectory_statistics.get("rStdDev"), 4),
        "thetaMin": round(trajectory_statistics.get("thetaMin"), 4),
        "thetaMax": round(trajectory_statistics.get("thetaMax"), 4),
        "thetaMean": round(trajectory_statistics.get("thetaMean"), 4),
        "thetaStdDev": round(trajectory_statistics.get("thetaStdDev"), 4),
        "vMin": round(trajectory_statistics.get("vMin"), 4),
        "vMax": round(trajectory_statistics.get("vMax"), 4),
        "vMean": round(trajectory_statistics.get("vMean"), 4),
        "vStdDev": round(trajectory_statistics.get("vStdDev"), 4),
    }
