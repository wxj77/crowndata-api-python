from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator

from crowndata_evaluation.routers.v1.helper import JointData, PoseData
from crowndata_evaluation.services.action_consistency.action_variance_calculator import (
    ActionVarianceCalculator,
)
from crowndata_evaluation.services.shape.geometry import calculate_trajectory_statistics
from crowndata_evaluation.services.utils import (
    fetch_trajectory_json,
    fetch_trajectory_sample_rate,
)

metric_router = APIRouter()


# Request model
class EvaluationMetricRequest(BaseModel):
    dataList: List[JointData] = Field(
        None,
        example=[
            {
                "data": [
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                    [0.1, 1.1, 2.1, 3.1, 4.1, 5.1],
                ],
                "sample_rate": 10.0,
                "name": "cartesian_position",
            }
        ],
    )
    dataName: PoseData = Field(
        None,
        example={"dataName": "droid_00000000", "joints": ["cartesian_position"]},
    )

    # Custom validator to ensure each inner list has exactly 6 elements
    @validator("dataList")
    def check_dataList_length(cls, data_list):
        if data_list is not None:
            for v in data_list:
                for inner_list in v.data:
                    if len(inner_list) != 6 and len(inner_list) != 7:
                        raise HTTPException(
                            status_code=400,
                            detail="Data validation error: each inner list must have exactly 6 or 7 elements.",
                        )
            return data_list


# Response model
class EvaluationMetric(BaseModel):
    name: Optional[str]
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


class EvaluationMetricResponse(BaseModel):
    evaluationMetric: List[EvaluationMetric]


# POST endpoint for evaluating metrics
@metric_router.post(
    "",
    summary="Metric",
    description="Retrieve metric",
    response_model=EvaluationMetricResponse,
)
async def post(request: EvaluationMetricRequest):
    # Check if both 'dataList' and 'dataName' are provided or neither is
    if (request.dataList is not None and request.dataName is not None) or (
        request.dataList is None and request.dataName is None
    ):
        # Raise an HTTPException with a 400 status code for bad request
        raise HTTPException(
            status_code=400,
            detail="Provide either 'dataList' or 'dataName', not both or neither.",
        )

    data_list = []
    if request.dataList is not None:
        data_list = request.dataList
        for v in data_list:
            if len(v.data[0]) == 7:
                dt = v.data[0][:, 6]
                sample_rate = 1.0 / dt
                v.sample_rate = sample_rate
            v.data = np.array(v.data)
    elif request.dataName is not None:
        for k in request.dataName.joints:
            data = fetch_trajectory_json(data_name=request.dataName.dataName, joint=k)
            sample_rate = fetch_trajectory_sample_rate(
                data_name=request.dataName.dataName
            )
            joint_data = JointData(data=data, sample_rate=sample_rate, name=k)
            data_list.append(joint_data)

    evaluation_metric = []

    for v in data_list:
        avc = ActionVarianceCalculator(r=0.1)
        action_consistency = avc.calculate_action_variance(v.data)
        sample_rate = v.sample_rate

        xyz_array = v.data[:, :3]
        trajectory_statistics = calculate_trajectory_statistics(
            xyz_array=xyz_array,
            sample_rate=sample_rate,
        )

        evaluation_metric.append(
            {
                "name": v.name,
                "actionConsistency": float(round(action_consistency, 4)),
                "curveLength": float(
                    round(trajectory_statistics.get("curveLength"), 4)
                ),
                "xMin": float(round(trajectory_statistics.get("xMin"), 4)),
                "xMax": float(round(trajectory_statistics.get("xMax"), 4)),
                "xMean": float(round(trajectory_statistics.get("xMean"), 4)),
                "xStdDev": float(round(trajectory_statistics.get("xStdDev"), 4)),
                "yMin": float(round(trajectory_statistics.get("yMin"), 4)),
                "yMax": float(round(trajectory_statistics.get("yMax"), 4)),
                "yMean": float(round(trajectory_statistics.get("yMean"), 4)),
                "yStdDev": float(round(trajectory_statistics.get("yStdDev"), 4)),
                "zMin": float(round(trajectory_statistics.get("zMin"), 4)),
                "zMax": float(round(trajectory_statistics.get("zMax"), 4)),
                "zMean": float(round(trajectory_statistics.get("zMean"), 4)),
                "zStdDev": float(round(trajectory_statistics.get("zStdDev"), 4)),
                "rMin": float(round(trajectory_statistics.get("rMin"), 4)),
                "rMax": float(round(trajectory_statistics.get("rMax"), 4)),
                "rMean": float(round(trajectory_statistics.get("rMean"), 4)),
                "rStdDev": float(round(trajectory_statistics.get("rStdDev"), 4)),
                "thetaMin": float(round(trajectory_statistics.get("thetaMin"), 4)),
                "thetaMax": float(round(trajectory_statistics.get("thetaMax"), 4)),
                "thetaMean": float(round(trajectory_statistics.get("thetaMean"), 4)),
                "thetaStdDev": float(
                    round(trajectory_statistics.get("thetaStdDev"), 4)
                ),
                "vMin": float(round(trajectory_statistics.get("vMin"), 4)),
                "vMax": float(round(trajectory_statistics.get("vMax"), 4)),
                "vMean": float(round(trajectory_statistics.get("vMean"), 4)),
                "vStdDev": float(round(trajectory_statistics.get("vStdDev"), 4)),
            }
        )

    return {"evaluationMetric": evaluation_metric}
