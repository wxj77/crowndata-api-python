# crowndata-api-python
Python API for “crowndata”

## API Documentation
The Crowndata Evaluation API provides interactive documentation generated automatically by FastAPI. You can access the docs via:

- Swagger UI: `${hostname}/docs/api/crowndata-api-python/crowndata-evaluation/`
- ReDoc: `${hostname}/redoc/api/crowndata-api-python/crowndata-evaluation/`

## How to compile

1. Install `poetry` with `pip3`

2. Install python dependencies `poetry install`

3. Run `poetry run uvicorn crowndata_evaluation.main:app --reload`


## How to format code
1. `poetry run black . && poetry run nbqa black .`
2. `poetry run autoflake --remove-all-unused-imports --recursive --in-place .`


