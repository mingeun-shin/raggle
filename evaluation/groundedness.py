import json
import re

from generation.llm import complete

CLAIM_CHECK_TEMPLATE = """\
You are verifying whether claims are supported by source passages.

Passages:
{contexts}

Claims:
{claims}

For each claim, decide if it is fully supported by the passages. A claim is \
supported only if the passages state or directly entail it. Partial support, \
plausible inference, or general-knowledge truth without passage support all \
count as unsupported.

Respond with only a JSON array, one object per claim, in order:
[{{"claim_index": 1, "supported": true}}, ...]"""


def _split_claims(answer_text: str) -> list[str]:
    # strip citation markers so the checker judges content, not formatting
    text = re.sub(r"\[\d+\]", "", answer_text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def _parse_json_array(raw: str) -> list[dict]:
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON array in judge output: {raw[:200]}")
    return json.loads(match.group())


def score_groundedness(answer_text: str, contexts: list[str]) -> dict:
    claims = _split_claims(answer_text)
    if not claims:
        # refusals / empty answers make no claims, so nothing is ungrounded
        return {"score": 1.0, "claims": []}

    prompt = CLAIM_CHECK_TEMPLATE.format(
        contexts="\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts)),
        claims="\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims)),
    )
    verdicts = _parse_json_array(complete(prompt))

    results = []
    for i, claim in enumerate(claims):
        verdict = next((v for v in verdicts if v.get("claim_index") == i + 1), None)
        # unparseable or missing verdict counts as unsupported: fail closed
        supported = bool(verdict and verdict.get("supported"))
        results.append({"claim": claim, "supported": supported})

    score = sum(r["supported"] for r in results) / len(results)
    return {"score": score, "claims": results}