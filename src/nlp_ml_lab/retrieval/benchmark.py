from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RetrievalCorpus:
    document_ids: list[str]
    texts: list[str]


@dataclass(frozen=True)
class RetrievalQueries:
    query_ids: list[str]
    texts: list[str]


@dataclass(frozen=True)
class RelevanceJudgements:
    query_to_documents: dict[str, set[str]]


def load_corpus_frame(frame: pd.DataFrame) -> RetrievalCorpus:
    required = {"_id", "text"}
    if not required.issubset(frame.columns):
        raise ValueError(f"corpus must contain {sorted(required)}")
    return RetrievalCorpus(
        document_ids=frame["_id"].astype(str).tolist(),
        texts=frame["text"].astype(str).tolist(),
    )


def load_queries_frame(frame: pd.DataFrame) -> RetrievalQueries:
    required = {"_id", "text"}
    if not required.issubset(frame.columns):
        raise ValueError(f"queries must contain {sorted(required)}")
    return RetrievalQueries(
        query_ids=frame["_id"].astype(str).tolist(),
        texts=frame["text"].astype(str).tolist(),
    )


def load_qrels_frame(frame: pd.DataFrame) -> RelevanceJudgements:
    required = {"query-id", "corpus-id", "score"}
    if not required.issubset(frame.columns):
        raise ValueError(f"qrels must contain {sorted(required)}")

    mapping: dict[str, set[str]] = {}
    for query_id, group in frame.groupby("query-id"):
        relevant = group.loc[group["score"] > 0, "corpus-id"].astype(str)
        mapping[str(query_id)] = set(relevant)

    return RelevanceJudgements(query_to_documents=mapping)
