"""LLM-as-judge correctness scoring vs reference answer, 1-5 rubric."""

def judge_answer(question: str, answer: str, reference: str) -> dict:
    """Returns {score: int, reasoning: str}."""