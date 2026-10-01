from dataclasses import dataclass

from nlp_ml_lab.retrieval.hybrid import reciprocal_rank_fusion
from nlp_ml_lab.retrieval.lexical import BM25Retriever
from nlp_ml_lab.retrieval.reranker import CrossEncoderReranker


@dataclass
class HybridSearchPipeline:
    document_ids: list[str]
    documents: list[str]
    lexical: BM25Retriever
    dense_search: object
    reranker: CrossEncoderReranker | None = None

    def search(
        self,
        query: str,
        *,
        candidate_k: int = 20,
        result_k: int = 10,
    ) -> list[tuple[str, float]]:
        if not query.strip():
            raise ValueError("query must not be blank")
        if candidate_k <= 0 or result_k <= 0:
            raise ValueError("candidate_k and result_k must be positive")

        lexical_results = self.lexical.search(query, k=candidate_k)
        dense_results = self.dense_search.search(query, k=candidate_k)

        lexical_ids = [item.document_id for item in lexical_results]
        dense_ids = [item[0] for item in dense_results]
        fused_ids = reciprocal_rank_fusion(
            [lexical_ids, dense_ids],
            limit=candidate_k,
        )

        if self.reranker is None:
            score_by_id = {
                item.document_id: item.score for item in lexical_results
            }
            score_by_id.update({item[0]: item[1] for item in dense_results})
            return [
                (document_id, float(score_by_id.get(document_id, 0.0)))
                for document_id in fused_ids[:result_k]
            ]

        document_by_id = dict(zip(self.document_ids, self.documents))
        pairs = [
            (document_id, document_by_id[document_id])
            for document_id in fused_ids
            if document_id in document_by_id
        ]
        return self.reranker.rerank(query, pairs, k=result_k)
