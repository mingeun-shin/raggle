def precision_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if k == 0 or not retrieved:
        return 0.0
    top = retrieved[:k]
    return sum(1 for cid in top if cid in gold) / len(top)


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    top = retrieved[:k]
    return sum(1 for cid in gold if cid in top) / len(gold)


def mrr(retrieved: list[str], gold: list[str]) -> float:
    for rank, cid in enumerate(retrieved, start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0