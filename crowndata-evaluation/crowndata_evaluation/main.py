from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from crowndata_evaluation.routers.v1.evaluation.metric_router import metric_router
from crowndata_evaluation.routers.v1.evaluation.compare_metric_router import (
    compare_metric_router,
)
from crowndata_evaluation.routers.v1.evaluation.group_metric_router import (
    group_metric_router,
)
from crowndata_evaluation.routers.v1.evaluation.group_compare_metric_router import (
    group_compare_metric_router,
)
from crowndata_evaluation.routers.v1.evaluation.compare_single_to_group_metric_route import (
    compare_single_to_group_router,
)
from crowndata_evaluation.routers.v1.data.information import (
    information_router,
)


app = FastAPI(
    title="Crowndata Evaluation API",
    description="API for evaluating metrics from Crowndata",
    version="1.0.0",
    docs_url="/docs/api/crowndata-api-python/crowndata-evaluation/",  # Custom path for Swagger UI
    redoc_url="/redoc/api/crowndata-api-python/crowndata-evaluation/",  # Custom path for ReDoc
)

# Allow all origins or specify domains like ['http://localhost:3000'] for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # Change "*" to specific domains in production for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(metric_router, prefix="/v1/evaluation/metrics", tags=["Metrics"])
app.include_router(
    compare_metric_router,
    prefix="/v1/evaluation/compare-metrics",
    tags=["Compare Metrics"],
)
app.include_router(
    group_metric_router,
    prefix="/v1/evaluation/group-metrics",
    tags=["Group Metrics"],
)
app.include_router(
    group_compare_metric_router,
    prefix="/v1/evaluation/group-compare-metrics",
    tags=["Group Compare Metrics"],
)
app.include_router(
    compare_single_to_group_router,
    prefix="/v1/evaluation/compare-single-to-group-metrics",
    tags=["Compare Single to Groups Metrics"],
)

app.include_router(
    information_router,
    prefix="/v1/data/information",
    tags=["Information"],
)
