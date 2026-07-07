"""Context assembly, generation, citation extraction."""
from dataclasses import dataclass
from ingestion.chunking import Chunk

@dataclass
class Answer:
    text: str
    citations: dict[int, str]   # citation number → chunk_id
    contexts: list[Chunk]

def generate_answer(query: str, chunks: list[Chunk]) -> Answer: ...