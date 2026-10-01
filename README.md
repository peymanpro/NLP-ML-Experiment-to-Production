# NLP-ML-Experiment-to-Production

A production-oriented NLP/ML engineering laboratory that takes two real information problems from data preparation to model selection, retrieval, serving, and operationalization.

The repository is intentionally **experiment-first and production-aware**. It does not hide model decisions behind a single library call, and it does not publish unverified benchmark numbers.

## What this project demonstrates

- Python data engineering and reproducibility
- scikit-learn classical ML baselines
- PyTorch dataset/training infrastructure
- Hugging Face Transformers and fine-tuning
- sentence embeddings
- dense vector search with FAISS
- BM25 lexical retrieval
- hybrid retrieval with Reciprocal Rank Fusion
- bi-encoder retrieval
- cross-encoder reranking
- classification and retrieval evaluation
- error analysis
- experiment provenance
- model artifact metadata and registry lifecycle
- FastAPI model serving
- health/readiness endpoints
- inference telemetry
- Docker deployment
- GitHub Actions quality gates

## The engineering question

The project uses the following decision rule:

> Start with the simplest defensible baseline. Add model complexity only when measured evidence justifies the additional quality, latency, memory, or operational cost.

This is why the project does not jump directly to a Transformer.

## Two connected problem tracks

### Track A — Intent Classification

BANKING77 provides 13,083 English customer-service queries across 77 fine-grained intent labels. The public Hugging Face dataset exposes the text and integer-label fields and provides train/test splits. citeturn366354search0turn366354search1

The implemented progression is:

~~~text
Raw query
   ↓
Validation / normalization
   ↓
TF-IDF
   ↓
Logistic Regression
   ↓
Reduced-dimensional tree baseline
   ↓
PyTorch mean-embedding model
   ↓
Pretrained Transformer
   ↓
Task fine-tuning
   ↓
Best-checkpoint selection
   ↓
Inference API
~~~

The source repository does not store the benchmark dataset or model weights.

### Track B — Retrieval and Ranking

The retrieval track uses the BEIR SciFact benchmark. The public dataset provides a scientific-claim retrieval corpus/query formulation and is currently listed under CC BY-SA 4.0. citeturn608861search0turn608861search1

The implemented progression is:

~~~text
Query
 ├── BM25
 └── Dense embedding
        ↓
 Candidate retrieval
        ↓
 Reciprocal Rank Fusion
        ↓
 Optional Cross-Encoder reranking
        ↓
 Recall@k / Precision@k / MRR / nDCG
~~~

The embedding reference model is sentence-transformers/all-MiniLM-L6-v2, an Apache-2.0 sentence-embedding model. citeturn366354search2

The reranking reference model is cross-encoder/ms-marco-MiniLM-L6-v2, an Apache-2.0 text-ranking model. citeturn366354search4

## Repository layout

~~~text
src/nlp_ml_lab/
├── data/
├── evaluation/
├── features/
├── mlops/
├── models/
├── observability/
├── retrieval/
├── serving/
├── text/
└── training/

tests/unit/
scripts/
docs/
.github/workflows/
~~~

## Architecture boundaries

~~~text
Data
 ↓
Modeling
 ↓
Evaluation
 ↓
Serving
 ↓
Operations
~~~

The repository keeps data acquisition, model implementation, evaluation logic, and serving contracts separate so each can be tested and evolved independently.

## Reproducibility

Experiments record enough context to reproduce a result:

~~~text
dataset identity
+ dataset version
+ dataset fingerprint
+ processing configuration
+ random seed
+ model configuration
+ code revision
~~~

Model artifacts can carry a manifest containing model version, task, dataset fingerprint, code revision, framework, and artifact path.

## Evaluation

Classification metrics:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1

Retrieval metrics:

- Recall@k
- Precision@k
- MRR
- nDCG@k

Operational evidence includes latency summaries and inference error counts.

## Serving

The FastAPI service exposes:

- GET /health
- GET /ready
- GET /metrics
- POST /predict

A model is loaded only from an explicit artifact path configured through environment variables. The service never silently trains a model at request time.

## Deployment

The repository includes:

- Dockerfile
- Docker Compose configuration
- environment example
- deployment documentation
- CI quality workflow
- lightweight smoke workflow

Build locally with:

~~~bash
docker build -t nlp-ml-experiment .
~~~

Run the application with:

~~~bash
python scripts/serve.py
~~~

See docs/DEPLOYMENT.md for model-artifact serving.

## Development

~~~bash
pip install -e ".[dev]"
python -m pytest
ruff check .
ruff format --check .
mypy src
~~~

Or use the Makefile:

~~~bash
make quality
make baseline
make pytorch
make transformer
make finetune
make retrieval
make serve
~~~

## Benchmark execution

Benchmark runners are deliberately separate from ordinary CI because they may download public datasets and model weights and can require substantial CPU/GPU time.

See:

- docs/BENCHMARK-RUNBOOK.md
- docs/EXPERIMENTS.md
- docs/RETRIEVAL.md
- docs/FINE-TUNING.md
- docs/MLOPS.md
- docs/RELEASE.md

No placeholder performance figures are presented as measured results.

## Portfolio role

This repository is the production bridge between several earlier first-principles projects.

~~~text
HowAttentionWorks
HowTransformersWork
HowLLMsWork
       ↓
conceptual understanding

Elasticsearch-Search-Platform
Evidence-Grounded-RAG
       ↓
search / retrieval engineering

NLP-ML-Experiment-to-Production
       ↓
data → baseline → PyTorch → Transformer
→ fine-tuning → retrieval → reranking
→ evaluation → API → Docker → monitoring
~~~

## Status

**Production-oriented v1 implementation complete.**

The codebase now contains the complete intended architecture for experimentation, model lifecycle, retrieval/ranking, serving, and operational controls.

External benchmark execution is intentionally a separate experiment step and is not represented by fabricated numbers in source control.

## License

MIT
