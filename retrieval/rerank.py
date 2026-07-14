from functools import lru_cache

from sentence_transformers import CrossEncoder

from ingestion.chunking import Chunk
from settings import settings


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    return CrossEncoder(settings.rerank_model)


def rerank(query: str, chunks: list[Chunk], top_m: int) -> list[Chunk]:
    if not chunks:
        return []
    scores = _model().predict([(query, c.text) for c in chunks])
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_m]]