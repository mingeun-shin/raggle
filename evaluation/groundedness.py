"""Claim-level faithfulness: each answer claim checked against contexts."""

def score_groundedness(answer_text: str, contexts: list[str]) -> dict:
    """Returns {score: float 0-1, claims: [{claim, supported}]}."""