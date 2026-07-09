from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from settings import settings


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(settings.embed_model)


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    return _model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 256,
    )


def embed_query(query: str) -> np.ndarray:
    # bge models want a query prefix; doc side is embedded bare
    return _model().encode(
        [f"Represent this sentence for searching relevant passages: {query}"],
        normalize_embeddings=True,
    )[0]