import json
import re

from generation.llm import complete

JUDGE_TEMPLATE = """\
You are grading an answer against a reference answer.

Question: {question}

Reference answer: {reference}

Candidate answer: {answer}

Grade the candidate on factual agreement with the reference:
5 - fully correct, covers all key points of the reference
4 - correct, misses minor details
3 - partially correct, misses a key point or adds minor errors
2 - mostly incorrect or missing most key points
1 - incorrect or contradicts the reference

Judge only factual content. Ignore style, length, phrasing, and citation \
markers. A candidate that correctly declines to answer when the reference \
contains a real answer scores 1.

Respond with only JSON: {{"score": <1-5>, "reasoning": "<one sentence>"}}"""


def judge_answer(question: str, answer: str, reference: str) -> dict:
    prompt = JUDGE_TEMPLATE.format(
        question=question, reference=reference, answer=answer
    )
    raw = complete(prompt, max_tokens=300)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON in judge output: {raw[:200]}")
    parsed = json.loads(match.group())
    score = int(parsed["score"])
    if not 1 <= score <= 5:
        raise ValueError(f"judge score out of range: {score}")
    return {"score": score, "reasoning": parsed.get("reasoning", "")}