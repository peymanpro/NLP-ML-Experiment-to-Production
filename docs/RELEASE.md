# Release Readiness

A release candidate is ready only when:

- unit tests pass;
- lint and formatting pass;
- type checking passes;
- dataset provenance is recorded;
- benchmark commands are reproducible;
- model artifacts contain provenance metadata;
- health/readiness behavior is defined;
- Docker build configuration is present;
- known limitations are documented.

## What is intentionally not included

Large datasets, model weights, caches, credentials, and environment-specific deployment configuration are not committed to the repository.

## Evidence policy

A benchmark number may be published only after the corresponding runner has actually executed and produced the result. No placeholder performance figure is presented as measured evidence.
