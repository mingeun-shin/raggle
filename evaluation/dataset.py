"""Eval dataset: JSONL of {question, gold_chunk_ids, reference_answer}."""
from dataclasses import dataclass

@dataclass
class EvalExample:
    question: str
    gold_chunk_ids: list[str]
    reference_answer: str

def load_dataset(path: str) -> list[EvalExample]: ...