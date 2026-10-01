# Benchmark Runbook

The project separates benchmark execution from CI because downloading public datasets and model weights is an experiment operation.

## Classification

Use:

~~~bash
python scripts/run_banking77_baseline.py --max-train 2500 --max-validation 500
python scripts/run_pytorch_baseline.py --max-train 2500 --max-validation 500
python scripts/run_transformer_baseline.py --max-train 500 --max-validation 200
python scripts/run_transformer_finetuning.py --max-train 500 --max-validation 200 --epochs 1
~~~

For a final benchmark, remove development sample limits and record the exact command/configuration.

## Retrieval

Use:

~~~bash
python scripts/run_scifact_retrieval.py --max-queries 300 --top-k 10
~~~

The retrieval runner uses explicit relevance judgements from the benchmark and does not generate its own labels.

## Artifact rule

Do not commit downloaded datasets, model weights, caches, or generated large artifacts.

Commit only small, human-readable benchmark summaries when they are intentionally selected as portfolio evidence.
