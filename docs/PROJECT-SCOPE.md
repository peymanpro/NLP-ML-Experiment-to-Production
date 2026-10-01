# Project Scope

## Primary objective

Build a reproducible NLP/ML engineering pipeline that demonstrates the progression from simple statistical baselines to Transformer, retrieval, ranking, and serving components.

## Track A — Intent classification

The first task is short natural-language intent classification using BANKING77 after verifying its upstream dataset metadata.

Pipeline:

~~~text
User Query → Preprocessing → Classifier → Intent
~~~

Models to compare:

1. TF-IDF + Logistic Regression
2. Pretrained Transformer
3. Fine-tuned Transformer

## Track B — Retrieval and ranking

The initial retrieval benchmark candidate is BEIR/SciFact. The exact dataset release, metadata, and licensing information must be verified during ingestion.

Pipeline:

~~~text
Query → Candidate Retrieval → Reranking → Ranked Results
~~~

The retrieval track will compare lexical, dense, hybrid, and second-stage reranking approaches.

## Non-goals

- training a large language model from scratch;
- building a generic chatbot;
- fabricating relevance labels;
- storing proprietary or sensitive datasets;
- presenting toy benchmark results as production performance.

## Decision rule

A more complex model is adopted only when evaluation evidence shows a meaningful benefit for the stated objective.

Each major model choice must record its baseline, expected benefit, evaluation metric, operational implications, and known failure modes.
