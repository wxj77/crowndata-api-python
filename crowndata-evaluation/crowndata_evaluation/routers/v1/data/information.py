from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from crowndata_evaluation.services.utils import fetch_information_json

information_router = APIRouter()


# Request model
class InformationRequest(BaseModel):
    dataName: str = Field(
        None,
        example="droid_00000000",
    )


# POST endpoint for evaluating metrics
@information_router.post(
    "",
    summary="Metric",
    description="Retrieve metric",
)
async def information(request: InformationRequest):
    # Check if both 'dataList' and 'dataName' are provided or neither is
    if request.dataName is None:
        # Raise an HTTPException with a 400 status code for bad request
        raise HTTPException(
            status_code=400,
            detail="Provide 'dataName'.",
        )

    return fetch_information_json(data_name=request.dataName)
