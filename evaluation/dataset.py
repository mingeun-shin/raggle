import json
from dataclasses import dataclass


@dataclass
class EvalExample:
    question: str
    gold_chunk_ids: list[str]
    reference_answer: str


def load_dataset(path: str) -> list[EvalExample]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {"question", "gold_chunk_ids", "reference_answer"} - row.keys()
            if missing:
                raise ValueError(f"line {i + 1}: missing fields {missing}")
            if not row["gold_chunk_ids"]:
                raise ValueError(f"line {i + 1}: empty gold_chunk_ids")
            examples.append(EvalExample(**{k: row[k] for k in
                ("question", "gold_chunk_ids", "reference_answer")}))
    if not examples:
        raise ValueError(f"empty dataset: {path}")
    return examples