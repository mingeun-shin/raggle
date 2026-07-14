ANSWER_TEMPLATE = """\
Answer the question using only the numbered context passages below.

Rules:
- Cite passages inline with bracketed numbers, e.g. [1] or [2][3].
- Every factual claim must carry at least one citation.
- If the passages do not contain the answer, say "I can't answer this from the provided documents." and nothing else.

Contexts:
{contexts}

Question: {question}

Answer:"""


def build_answer_prompt(question: str, contexts: list[str]) -> str:
    numbered = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return ANSWER_TEMPLATE.format(contexts=numbered, question=question)