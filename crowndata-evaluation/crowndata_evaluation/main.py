from fastapi import FastAPI
from crowndata_evaluation.routers.v1.evaluation.metric_router import metric_router
from crowndata_evaluation.routers.v1.evaluation.compare_metric_router import (
    compare_metric_router,
)
from crowndata_evaluation.routers.v1.evaluation.group_metric_router import (
    group_metric_router,
)

app = FastAPI(
    title="Crowndata Evaluation API",
    description="API for evaluating metrics from Crowndata",
    version="1.0.0",
    docs_url="/docs/api/crowndata-api-python/crowndata-evaluation/",  # Custom path for Swagger UI
    redoc_url="/redoc/api/crowndata-api-python/crowndata-evaluation/",  # Custom path for ReDoc
)

# Ensure that these two routers have different prefixes
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
