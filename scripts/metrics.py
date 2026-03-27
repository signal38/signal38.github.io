"""Evaluation metrics shared across notebooks and the app.

All metrics operate on escalation_level predictions (integers 1-5).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


# ── JSON output metrics ───────────────────────────────────────────────────

REQUIRED_FIELDS = [
    "escalation_level",
    "escalation_rationale",
    "situation_summary",
    "key_actors",
    "confidence_level",
]


def parse_json_output(text: str) -> dict | None:
    """Extract and parse JSON from model output. Returns None on failure."""
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def field_completeness(parsed: dict | None) -> float:
    """Fraction of required fields present in the parsed output."""
    if parsed is None:
        return 0.0
    present = sum(1 for f in REQUIRED_FIELDS if parsed.get(f) is not None)
    return present / len(REQUIRED_FIELDS)


# ── Escalation metrics ────────────────────────────────────────────────────

def escalation_mae(preds: list[int | None], targets: list[int]) -> float:
    """Mean absolute error on escalation level, ignoring None predictions."""
    pairs = [(p, t) for p, t in zip(preds, targets) if p is not None]
    if not pairs:
        return float("nan")
    return float(np.mean([abs(p - t) for p, t in pairs]))


def escalation_accuracy(preds: list[int | None], targets: list[int]) -> float:
    """Exact-match accuracy on escalation level."""
    pairs = [(p, t) for p, t in zip(preds, targets) if p is not None]
    if not pairs:
        return float("nan")
    return float(np.mean([p == t for p, t in pairs]))


# ── Naive baseline ────────────────────────────────────────────────────────

def goldstein_to_escalation(avg_goldstein: float) -> int:
    """Rule-based escalation level from Goldstein scale.

    Goldstein range: -10 (most conflictual) to +10 (most cooperative).
    NK military events are almost always negative; map to 1-5 risk scale.

      >= -2  → 1 (routine)
      >= -4  → 2 (elevated)
      >= -6  → 3 (significant)
      >= -8  → 4 (high)
       < -8  → 5 (critical)
    """
    if avg_goldstein >= -2:
        return 1
    elif avg_goldstein >= -4:
        return 2
    elif avg_goldstein >= -6:
        return 3
    elif avg_goldstein >= -8:
        return 4
    else:
        return 5


# ── Results serialization ─────────────────────────────────────────────────

def save_results(results: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(results, indent=2))
    print(f"Results saved to {path}")
