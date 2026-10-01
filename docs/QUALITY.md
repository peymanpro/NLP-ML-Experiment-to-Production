# Quality Gates

The repository treats code quality as part of the implementation, not as a post-hoc claim.

## Automated checks

Every push to `main` and every pull request runs:

1. editable package installation;
2. the full pytest suite;
3. Ruff linting;
4. Ruff format verification;
5. strict mypy type checking for the project source.

The workflow uses pip caching and cancels stale runs for the same Git ref. It does not modify the repository during CI.

## What CI does not claim

CI validates software quality gates. It does not manufacture dataset benchmark results, GPU training results, production traffic results, or latency figures.

Public benchmark commands remain separate so expensive experiments can be executed deliberately and recorded with their dataset, model, configuration, environment, and code revision.

## Evidence policy

A capability is described as implemented when the source code and automated tests establish the behavior.

A benchmark result is described as measured only when a benchmark runner has actually executed and produced the corresponding artifact.

A production deployment is described as reproducible when the deployment path, model-artifact contract, health/readiness behavior, and operational configuration are documented and tested.
