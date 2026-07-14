import re

from rank_bm25 import BM25Okapi

import indexing.chunk_store as chunk_store

_index: BM25Okapi | None = None
_chunk_ids: list[str] = []


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def build() -> None:
    global _index, _chunk_ids
    chunks = chunk_store.load_all_chunks()
    if not chunks:
        _index, _chunk_ids = None, []
        return
    _chunk_ids = [c.chunk_id for c in chunks]
    _index = BM25Okapi([_tokenize(c.text) for c in chunks])


def search(query: str, top_k: int) -> list[tuple[str, float]]:
    if _index is None:
        build()
    if _index is None:
        return []
    scores = _index.get_scores(_tokenize(query))
    ranked = sorted(zip(_chunk_ids, scores), key=lambda x: x[1], reverse=True)
    return [(cid, s) for cid, s in ranked[:top_k] if s > 0]