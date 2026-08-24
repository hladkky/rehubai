"""Execution-quality functional Q.

Implements the objective from ``MainAlgorithm.md``::

    Q = n_cycles / ( n_cycles + w_v * |V| + w_t * sum |tau_ij - tau_ij^opt| )

Q rewards completed cycles and penalises violations and off-tempo transitions. It
is the single scalar summarising a session and, once calibrated against clinical
scores (KIMORE), becomes an interpretable quality metric.
"""

from __future__ import annotations

from typing import Tuple


def compute_quality(
    n_cycles: int,
    n_violations: int,
    transitions: int = 0,
    tempo_penalty: float = 0.0,
    weights: Tuple[float, float] = (1.0, 0.5),
) -> float:
    """Compute Q in ``[0, 1]``.

    Args:
        n_cycles: Number of successfully completed exercise cycles.
        n_violations: Total number of violations |V| over the session.
        transitions: Number of transitions actually performed (reserved for
            future tempo weighting; unused when ``tempo_penalty`` is given directly).
        tempo_penalty: Aggregated tempo deviation ``sum |tau - tau_opt|`` in seconds.
        weights: ``(w_v, w_t)`` weighting of violations and tempo penalty.

    Returns:
        Quality score in ``[0, 1]``. Returns ``0.0`` when no cycle was completed
        (nothing valid to score).
    """
    w_v, w_t = weights
    if n_cycles <= 0:
        return 0.0
    denom = n_cycles + w_v * max(n_violations, 0) + w_t * max(tempo_penalty, 0.0)
    if denom <= 0:
        return 0.0
    return float(n_cycles) / float(denom)
