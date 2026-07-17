import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone

import pipeline
from evaluation.dataset import load_dataset
from evaluation.groundedness import score_groundedness
from evaluation.judge import judge_answer
from evaluation.retrieval_metrics import mrr, precision_at_k, recall_at_k
from settings import settings


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_eval(dataset_path: str = "eval_data/dataset.jsonl") -> dict:
    start = time.monotonic()
    examples = load_dataset(dataset_path)
    k = settings.rerank_top_m
    per_example = []

    for ex in examples:
        retrieved = pipeline.retrieved_ids(ex.question)
        answer = pipeline.query(ex.question)

        row = {
            "question": ex.question,
            "retrieved_ids": retrieved,
            "gold_chunk_ids": ex.gold_chunk_ids,
            "answer": answer.text,
            "precision_at_k": precision_at_k(retrieved, ex.gold_chunk_ids, k),
            "recall_at_k": recall_at_k(retrieved, ex.gold_chunk_ids, k),
            "mrr": mrr(retrieved, ex.gold_chunk_ids),
        }

        try:
            grounded = score_groundedness(
                answer.text, [c.text for c in answer.contexts]
            )
            row["groundedness"] = grounded["score"]
            row["claims"] = grounded["claims"]
        except (ValueError, json.JSONDecodeError) as e:
            row["groundedness"] = None
            row["groundedness_error"] = str(e)

        try:
            verdict = judge_answer(ex.question, answer.text, ex.reference_answer)
            row["judge_score"] = verdict["score"]
            row["judge_reasoning"] = verdict["reasoning"]
        except (ValueError, json.JSONDecodeError, KeyError) as e:
            row["judge_score"] = None
            row["judge_error"] = str(e)

        per_example.append(row)

    scored_g = [r["groundedness"] for r in per_example if r["groundedness"] is not None]
    scored_j = [r["judge_score"] for r in per_example if r["judge_score"] is not None]
    errors = len(per_example) - min(len(scored_g), len(scored_j))

    report = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "config": asdict(settings),
        "config_hash": settings.config_hash(),
        "dataset": dataset_path,
        "n_examples": len(examples),
        "n_scoring_errors": errors,
        "metrics": {
            "precision_at_k": _mean([r["precision_at_k"] for r in per_example]),
            "recall_at_k": _mean([r["recall_at_k"] for r in per_example]),
            "mrr": _mean([r["mrr"] for r in per_example]),
            "groundedness": _mean(scored_g),
            "judge_score": _mean(scored_j),
        },
        "per_example": per_example,
        "duration_s": None,
    }

    report["duration_s"] = round(time.monotonic() - start, 2)
    os.makedirs(settings.runs_dir, exist_ok=True)
    path = os.path.join(settings.runs_dir, f"{report['run_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report