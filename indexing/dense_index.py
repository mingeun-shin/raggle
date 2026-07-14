from functools import lru_cache

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from settings import settings

_COLLECTION = "chunks"


@lru_cache(maxsize=1)
def _client() -> QdrantClient:
    return QdrantClient(path=f"{settings.data_dir}/qdrant")


def _ensure_collection(dim: int) -> None:
    client = _client()
    if not client.collection_exists(_COLLECTION):
        client.create_collection(
            _COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def upsert(chunk_ids: list[str], vectors: np.ndarray) -> None:
    _ensure_collection(vectors.shape[1])
    points = [
        PointStruct(id=i, vector=vec.tolist(), payload={"chunk_id": cid})
        for i, (cid, vec) in enumerate(zip(chunk_ids, vectors))
    ]
    _client().upsert(_COLLECTION, points=points)


def search(query_vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
    hits = _client().query_points(
        _COLLECTION, query=query_vector.tolist(), limit=top_k
    ).points
    return [(h.payload["chunk_id"], h.score) for h in hits]


def clear() -> None:
    client = _client()
    if client.collection_exists(_COLLECTION):
        client.delete_collection(_COLLECTION)