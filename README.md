# NLP-ML-Experiment-to-Production

A production-oriented NLP/ML engineering project that traces a machine-learning problem from dataset definition and classical baselines through PyTorch, Transformers, semantic retrieval, ranking, evaluation, and deployment.

The project is intentionally an experiment-to-production pipeline rather than a model-only demo.

## Problem tracks

Track A is intent classification over short natural-language support queries.

Track B is semantic retrieval and ranking over a public information-retrieval benchmark.

Both tracks share reproducibility, evaluation, model-selection, serving, and operational infrastructure.

## Core question

When does additional model complexity produce enough measurable value to justify its cost?

## Architecture

~~~text
Problem Definition
      ↓
Dataset Provenance
      ↓
Data Quality
      ↓
Classical Baselines
      ↓
PyTorch Training
      ↓
Transformer Baseline
      ↓
Fine-Tuning
      ↓
Embedding / Retrieval
      ↓
Ranking / Reranking
      ↓
Evaluation
      ↓
Model Selection
      ↓
FastAPI
      ↓
Docker / CI
      ↓
Monitoring
~~~

## Engineering principles

- Establish a simple baseline before introducing a complex model.
- Keep data, modeling, evaluation, and serving concerns separate.
- Treat measurements and failure cases as first-class artifacts.
- Make experiments reproducible.
- Do not present toy benchmark results as production performance.

## Development

Python 3.12 is the target runtime.

Typical quality gates:

~~~bash
python -m pytest
ruff check .
ruff format --check .
mypy src
~~~

Downloaded datasets and trained model weights are intentionally excluded from source control.

Project execution status is maintained in the private control repository:
peymanpro/Git-Projects-Phase
