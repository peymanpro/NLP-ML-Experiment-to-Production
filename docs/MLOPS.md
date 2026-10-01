# MLOps and Model Lifecycle

The project treats a trained model as a versioned artifact with provenance.

## Lifecycle

Candidate → Staging → Production → Archived

Each model record includes:

- model name/version;
- task;
- artifact location;
- dataset fingerprint;
- source revision;
- evaluation metrics;
- lifecycle stage.

## Reproducibility

The minimum provenance chain is:

Dataset identity
+
Processing configuration
+
Random seed
+
Model configuration
+
Code revision

## Deployment

Model artifacts are loaded from an explicit filesystem path at serving time. The serving process does not silently train or modify a model.

## Health and readiness

The health endpoint answers whether the process is alive.

The readiness endpoint answers whether the model is available for inference.

## Observability

The service records request count, error count, and average inference latency.

This is intentionally lightweight. A production deployment can replace the internal metrics implementation with Prometheus or OpenTelemetry instrumentation without changing the inference contract.
