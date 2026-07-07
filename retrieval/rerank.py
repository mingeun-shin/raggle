"""Cross-encoder reranking of fused candidates."""
from ingestion.chunking import Chunk

def rerank(query: str, chunks: list[Chunk], top_m: int) -> list[Chunk]: ...