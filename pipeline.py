"""Top-level orchestration. API and eval runner both call through here."""
from generation.answer import Answer

def ingest(corpus_dir: str) -> dict: ...
def query(question: str) -> Answer: ...
def retrieved_ids(question: str) -> list[str]:
    """Retrieval-only path (post-rerank chunk ids) for eval."""