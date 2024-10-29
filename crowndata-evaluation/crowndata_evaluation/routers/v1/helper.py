from typing import List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field

from crowndata_evaluation.services.utils import default_sample_rate


class PoseData(BaseModel):
    dataName: str = Field(None, example="droid_00000000")
    joints: List[str] = Field(None, example=["cartesian_position"])


class JointData(BaseModel):
    name: Optional[str] = Field(
        "example_joint",
        example="example_joint",
        description="joint name",
    )
    data: Union[List[List[float]], np.ndarray] = Field(
        None,
        example=[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ],
    )
    sample_rate: Optional[float] = Field(
        default_sample_rate,
        example=default_sample_rate,
        description="Sampling rate in Hz, must be a positive number.",
    )

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {np.ndarray: lambda v: v.tolist()}

    def __init__(self, **data):
        super().__init__(**data)
        # Set sample rate to default if not provided
        if self.sample_rate is None:
            self.sample_rate = default_sample_rate

        # Convert data to numpy array if it is not already one
        if isinstance(self.data, list):
            self.data = np.array(self.data)
