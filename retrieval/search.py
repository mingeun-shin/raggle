import indexing.dense_index as dense_index
import indexing.sparse_index as sparse_index
from indexing.embedder import embed_query
from settings import settings


def _rrf(rankings: list[list[str]], k: int) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def hybrid_search(query: str) -> list[str]:
    dense = [cid for cid, _ in dense_index.search(embed_query(query), settings.dense_top_k)]
    sparse = [cid for cid, _ in sparse_index.search(query, settings.sparse_top_k)]
    fused = _rrf([dense, sparse], k=settings.rrf_k)
    return fused[: settings.fusion_top_n]