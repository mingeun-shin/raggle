import os
from dataclasses import dataclass, field, asdict
import hashlib, json

@dataclass(frozen=True)
class Settings:
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "claude-sonnet-4-6"

    chunk_max_tokens: int = 350
    chunk_overlap_tokens: int = 50

    dense_top_k: int = 20
    sparse_top_k: int = 20
    rrf_k: int = 60
    fusion_top_n: int = 12
    rerank_top_m: int = 5

    data_dir: str = "data"
    runs_dir: str = "runs"
    baselines_dir: str = "baselines"

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]

settings = Settings()