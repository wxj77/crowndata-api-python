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
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, z)

    # Compute the differences in position and time (dx, dy, dz, dt)
    dx, dy, dz = np.diff(x), np.diff(y), np.diff(z)

    # Compute velocity components and magnitude
    if dt is not None:
        vx, vy, vz = dx / dt, dy / dt, dz / dt
    elif sample_rate is not None:
        vx, vy, vz = dx * sample_rate, dy * sample_rate, dz * sample_rate

    v = np.sqrt(vx**2 + vy**2 + vz**2)

    x_statistics = compute_statistics(x)
    x_min, x_max, x_mean, x_std_dev = (
        x_statistics.get("min"),
        x_statistics.get("max"),
        x_statistics.get("mean"),
        x_statistics.get("std_dev"),
    )
    y_statistics = compute_statistics(y)
    y_min, y_max, y_mean, y_std_dev = (
        y_statistics.get("min"),
        y_statistics.get("max"),
        y_statistics.get("mean"),
        y_statistics.get("std_dev"),
    )
    z_statistics = compute_statistics(z)
    z_min, z_max, z_mean, z_std_dev = (
        x_statistics.get("min"),
        z_statistics.get("max"),
        z_statistics.get("mean"),
        z_statistics.get("std_dev"),
    )
    r_statistics = compute_statistics(r)
    r_min, r_max, r_mean, r_std_dev = (
        r_statistics.get("min"),
        r_statistics.get("max"),
        r_statistics.get("mean"),
        r_statistics.get("std_dev"),
    )
    theta_statistics = compute_statistics(theta)
    theta_min, theta_max, theta_mean, theta_std_dev = (
        theta_statistics.get("min"),
        theta_statistics.get("max"),
        theta_statistics.get("mean"),
        theta_statistics.get("std_dev"),
    )
    v_statistics = compute_statistics(v)
    v_min, v_max, v_mean, v_std_dev = (
        v_statistics.get("min"),
        v_statistics.get("max"),
        v_statistics.get("mean"),
        v_statistics.get("std_dev"),
    )

    # Curve Length
    curve_length = calculate_curve_length_3d(xyz_array)

    return {
        "actionConsistency": round(action_consistency, 4),
        "curveLength": round(curve_length, 4),
        "xMin": round(x_min, 4),
        "xMax": round(x_max, 4),
        "xMean": round(x_mean, 4),
        "xStdDev": round(x_std_dev, 4),
        "yMin": round(y_min, 4),
        "yMax": round(y_max, 4),
        "yMean": round(y_mean, 4),
        "yStdDev": round(y_std_dev, 4),
        "zMin": round(z_min, 4),
        "zMax": round(z_max, 4),
        "zMean": round(z_mean, 4),
        "zStdDev": round(z_std_dev, 4),
        "rMin": round(r_min, 4),
        "rMax": round(r_max, 4),
        "rMean": round(r_mean, 4),
        "rStdDev": round(r_std_dev, 4),
        "thetaMin": round(theta_min, 4),
        "thetaMax": round(theta_max, 4),
        "thetaMean": round(theta_mean, 4),
        "thetaStdDev": round(theta_std_dev, 4),
        "vMin": round(v_min, 4),
        "vMax": round(v_max, 4),
        "vMean": round(v_mean, 4),
        "vStdDev": round(v_std_dev, 4),
    }
