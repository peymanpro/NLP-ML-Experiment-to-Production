# V1 Implementation Status

## Scope

Version 1 covers the complete technical path defined for this repository:

1. reproducible dataset acquisition and provenance;
2. data validation and quality inspection;
3. classical text classification baselines;
4. PyTorch Dataset, batching, training, evaluation, and checkpoints;
5. Hugging Face tokenizer and Transformer classification;
6. direct PyTorch fine-tuning with validation-based checkpoint selection;
7. experiment and model provenance records;
8. dense embeddings and FAISS retrieval;
9. BM25 lexical retrieval;
10. reciprocal-rank-fusion hybrid retrieval;
11. bi-encoder retrieval;
12. cross-encoder reranking;
13. retrieval and classification metrics;
14. FastAPI inference contracts and endpoints;
15. model loading from explicit artifacts;
16. model registry lifecycle;
17. latency/error telemetry;
18. lightweight input-distribution drift measurement;
19. Docker and Docker Compose serving;
20. GitHub Actions quality and smoke workflows.

## Evidence policy

The source repository intentionally does not contain fabricated benchmark results.

Public datasets and model weights are downloaded by explicit experiment runners. Their outputs become portfolio evidence only after an actual run produces the result.

## Verification policy

The final source revision must pass:

- pytest;
- Ruff linting;
- Ruff formatting check;
- mypy.

Benchmark execution remains separate from CI because it may download large datasets/model weights and require substantial compute.

## Current state

The repository contains the complete V1 implementation. The final CI verification is the remaining release gate for this revision.
