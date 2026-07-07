"""Diff run vs baseline with per-metric absolute thresholds."""

def compare(current: dict, baseline: dict) -> dict:
    """Returns {passed: bool, deltas: {...}, failures: [...]}."""