# Dataset Strategy

## Classification

BANKING77 is the initial classification benchmark because it provides a realistic multi-class intent-routing task with short natural-language queries.

The dataset itself is not committed to source control.

The ingestion record must capture:

- dataset identifier;
- upstream source;
- version or release information;
- license;
- split structure;
- retrieval date;
- local fingerprint.

## Retrieval

The retrieval track will begin with a public benchmark such as BEIR/SciFact after explicit verification of its exact release metadata and license.

Original relevance semantics must be preserved.

## Governance

No benchmark label may be silently rewritten.

Any filtering, normalization, deduplication, or sampling must be explicit and reproducible.

## Leakage prevention

The test split must remain isolated from training and model-selection decisions.

## Reproducibility

A dataset run should be traceable to:

~~~text
Dataset identity
+ Dataset version
+ Processing configuration
+ Random seed
+ Code revision
~~~
