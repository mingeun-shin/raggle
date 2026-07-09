import json
import os

from ingestion.chunking import Chunk
from settings import settings

_PATH = os.path.join(settings.data_dir, "chunks", "chunks.jsonl")


def save_chunks(chunks: list[Chunk]) -> None:
    os.makedirs(os.path.dirname(_PATH), exist_ok=True)
    with open(_PATH, "a", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")


def load_all_chunks() -> list[Chunk]:
    if not os.path.exists(_PATH):
        return []
    with open(_PATH, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


def get_chunks(chunk_ids: list[str]) -> list[Chunk]:
    by_id = {c.chunk_id: c for c in load_all_chunks()}
    missing = [cid for cid in chunk_ids if cid not in by_id]
    if missing:
        raise KeyError(f"chunk ids not in store: {missing}")
    return [by_id[cid] for cid in chunk_ids]


def clear() -> None:
    if os.path.exists(_PATH):
        os.remove(_PATH)