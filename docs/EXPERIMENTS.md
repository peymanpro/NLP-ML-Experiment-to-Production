# Experiment Methodology

Every material model change should answer five questions:

1. What is the baseline?
2. What changed?
3. What metric should improve?
4. What operational cost changed?
5. What failure modes changed?

## Classification

The primary metric is macro F1 because the BANKING77 task contains 77 intents and the project cares about performance across classes, not only aggregate accuracy. BANKING77 is an English 77-intent classification benchmark with 13,083 queries. citeturn366354search0turn366354search1

Secondary metrics include accuracy, macro precision, and macro recall.

## Retrieval

The project reports Recall@k, Precision@k, MRR, and nDCG@k.

## Controlled comparisons

When comparing two models:

- keep the dataset split fixed;
- keep evaluation data fixed;
- record the random seed;
- record model and tokenizer revisions;
- record relevant hyperparameters;
- avoid using the final test set for model selection.

## Operational evidence

Where applicable, record inference latency, model artifact size, memory/compute requirements, retrieval index size, and failure behavior.

No benchmark result is copied into documentation unless a reproducible run produced it.
