import re
from dataclasses import dataclass
from functools import lru_cache

from transformers import AutoTokenizer

from ingestion.loaders import Document
from settings import settings


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    position: int
    metadata: dict


@lru_cache(maxsize=1)
def _tokenizer():
    # same tokenizer the embedder truncates with, so chunk budgets are honest
    return AutoTokenizer.from_pretrained(settings.embed_model)


def _n_tokens(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False))


def _split_recursive(text: str, max_tokens: int) -> list[str]:
    if _n_tokens(text) <= max_tokens:
        return [text]

    # try paragraph, then sentence, then hard word split
    for pattern in (r"\n\n+", r"(?<=[.!?])\s+"):
        parts = [p for p in re.split(pattern, text) if p.strip()]
        if len(parts) > 1:
            pieces = []
            for part in parts:
                pieces.extend(_split_recursive(part, max_tokens))
            return pieces

    words = text.split()
    mid = len(words) // 2
    return _split_recursive(" ".join(words[:mid]), max_tokens) + _split_recursive(
        " ".join(words[mid:]), max_tokens
    )


def _pack(pieces: list[str], max_tokens: int, overlap_tokens: int) -> list[str]:
    """Greedily merge small pieces up to the budget, carrying tail overlap."""
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for piece in pieces:
        pt = _n_tokens(piece)
        if current and current_tokens + pt > max_tokens:
            chunks.append(" ".join(current))
            # seed next chunk with trailing pieces up to the overlap budget
            tail: list[str] = []
            tail_tokens = 0
            for prev in reversed(current):
                t = _n_tokens(prev)
                if tail_tokens + t > overlap_tokens:
                    break
                tail.insert(0, prev)
                tail_tokens += t
            current, current_tokens = tail, tail_tokens
        current.append(piece)
        current_tokens += pt

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(
    doc: Document,
    max_tokens: int = settings.chunk_max_tokens,
    overlap_tokens: int = settings.chunk_overlap_tokens,
) -> list[Chunk]:
    pieces = _split_recursive(doc.text, max_tokens)
    texts = _pack(pieces, max_tokens, overlap_tokens)
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}-{i:04d}",
            doc_id=doc.doc_id,
            text=text,
            position=i,
            metadata=dict(doc.metadata),
        )
        for i, text in enumerate(texts)
    ]