"""Violation interpretation → prioritised, de-duplicated user feedback.

Turns the raw violation dicts emitted by :class:`~rehubai.validator.ExerciseValidator`
into a short, ranked list of corrective messages suitable for real-time display.
Ordering surfaces the most actionable issue first: a wrong stage order (SEQUENTIAL)
matters more than a slightly-off angle, and larger angle deviations rank above
smaller ones. (Closes the "error-interpretation" item in the project TODO.)
"""

from __future__ import annotations

from typing import List, Sequence

from .validator import DYNAMIC, GEOMETRIC, GLOBAL, SEQUENTIAL

# Lower number = higher priority (shown first).
_PRIORITY = {
    SEQUENTIAL: 0,
    DYNAMIC: 1,
    GEOMETRIC: 2,
    GLOBAL: 3,
}


def _rank(v: dict) -> tuple:
    """Sort key: (category priority, -deviation magnitude)."""
    prio = _PRIORITY.get(v.get("type"), 9)
    magnitude = abs(float(v.get("deviation", 0.0)))
    return (prio, -magnitude)


def interpret_violations(
    violations: Sequence[dict], top_k: int = 3
) -> List[str]:
    """Return up to ``top_k`` distinct feedback messages, most important first.

    Args:
        violations: Raw violation dicts (each may carry ``type``/``constraint``/
            ``deviation``/``feedback``).
        top_k: Maximum number of messages to return.

    Returns:
        Ordered, de-duplicated list of feedback strings.
    """
    ranked = sorted(
        (v for v in violations if v.get("feedback")),
        key=_rank,
    )
    seen = set()
    out: List[str] = []
    for v in ranked:
        key = (v.get("type"), v.get("constraint"))
        if key in seen:
            continue
        seen.add(key)
        out.append(v["feedback"])
        if len(out) >= top_k:
            break
    return out
