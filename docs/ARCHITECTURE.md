# System Architecture

The repository is organized around four boundaries.

## 1. Data

Responsible for acquisition, provenance, validation, splitting, and quality statistics.

## 2. Modeling

Responsible for classical baselines, PyTorch neural baseline, Transformer baseline, fine-tuning, embedding models, retrieval, and reranking.

## 3. Evaluation

Responsible for classification metrics, retrieval metrics, error analysis, experiment records, and model selection.

## 4. Serving and Operations

Responsible for prediction contracts, HTTP serving, model artifacts, model registry, health/readiness, latency/error telemetry, and Docker deployment.

The intended dependency direction is:

Data → Modeling → Evaluation → Serving

Evaluation must not depend on the HTTP layer, and serving should consume explicit model contracts rather than reach into training internals.
