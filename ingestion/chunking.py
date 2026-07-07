"""Recursive token-bounded splitting with overlap."""
from dataclasses import dataclass
from ingestion.loaders import Document

@dataclass
class Chunk:
    chunk_id: str       # deterministic: doc_id + position
    doc_id: str
    text: str
    position: int
    metadata: dict

def chunk_document(doc: Document, max_tokens: int, overlap_tokens: int) -> list[Chunk]: ...