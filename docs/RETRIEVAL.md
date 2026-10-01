
# Retrieval and Ranking

The retrieval track uses public relevance judgements rather than manually invented labels.

The initial benchmark is BEIR/SciFact. The public Hugging Face dataset describes separate corpus and query subsets and the benchmark is intended for scientific-claim retrieval. The dataset card currently lists CC BY-SA 4.0 licensing. The relevance judgements remain separate so the evaluation semantics are explicit. citeturn608861search0turn608861search1

## Pipeline

Query
  ↓
BM25
  +
Dense Embedding
  ↓
Hybrid Candidate Set
  ↓
Bi-Encoder Retrieval
  ↓
Cross-Encoder Reranking
  ↓
Evaluation

The dense embedding reference model is sentence-transformers/all-MiniLM-L6-v2, an Apache-2.0 model intended for sentence embeddings. citeturn366354search2

The reranking reference model is cross-encoder/ms-marco-MiniLM-L6-v2, an Apache-2.0 cross-encoder model intended for text ranking. citeturn366354search4

The project reports Recall@k, Precision@k, MRR, and nDCG@k.

The benchmark runner is deliberately outside ordinary unit-test execution because downloading a model and a retrieval corpus is an experiment operation, not a test prerequisite.
