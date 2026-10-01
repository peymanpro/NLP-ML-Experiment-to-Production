# Release Readiness

## V1 release candidate

The declared V1 implementation is complete and has passed the repository quality gates.

Verified implementation revision:

`f8ea33d35862310ab853c065a8664d3527cdbfb8`

Verification evidence:

- quality workflow: **success** — run `36886823091`;
- smoke workflow: **success** — run `36886823125`;
- automated test suite: **94 passed**;
- Ruff lint: **passed**;
- Ruff format check: **passed**;
- mypy: **passed**.

## Release contents

The release candidate includes:

- reproducible dataset loading and provenance;
- classical ML baselines;
- PyTorch training and checkpointing;
- Hugging Face Transformer baseline and fine-tuning;
- dense, lexical, hybrid, bi-encoder and cross-encoder retrieval/ranking;
- evaluation and error analysis utilities;
- experiment and model provenance;
- model registry lifecycle;
- FastAPI serving with health/readiness/metrics/predict endpoints;
- Docker and Docker Compose deployment;
- telemetry and input-distribution drift measurement;
- CI quality and smoke checks;
- implementation and operational documentation.

## Evidence boundary

Large datasets, model weights, caches, credentials, and environment-specific deployment configuration are not committed.

Benchmark numbers are published only after the relevant benchmark runner has actually executed and produced the result. The repository does not treat the existence of a runner as evidence that an experiment was completed.
