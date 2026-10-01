# Deployment Guide

## Local Python execution

Install the package:

~~~bash
pip install .
~~~

Run the API:

~~~bash
python scripts/serve.py
~~~

The API exposes health and readiness endpoints. Prediction remains unavailable until NLP_ML_MODEL_PATH points to a compatible Hugging Face sequence-classification artifact.

## Docker

Build:

~~~bash
docker build -t nlp-ml-experiment .
~~~

Run with a local model artifact:

~~~bash
docker run --rm -p 8000:8000   -e NLP_ML_MODEL_PATH=/app/model   -v "$PWD/models/serving:/app/model:ro"   nlp-ml-experiment
~~~

Docker Compose is provided for the same local pattern.

## Production considerations

A real production deployment should additionally provide external configuration/secret management, TLS termination, resource limits, autoscaling where required, centralized telemetry, authenticated access to prediction endpoints where required, and artifact storage outside the container image for large models.

Those concerns are deployment-environment responsibilities and are not hidden by this repository.
