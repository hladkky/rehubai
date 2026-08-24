"""Unit tests for the quality functional and feedback interpretation."""

from rehubai.feedback import interpret_violations
from rehubai.quality import compute_quality
from rehubai.validator import DYNAMIC, GEOMETRIC, GLOBAL, SEQUENTIAL


def test_quality_perfect_when_no_violations():
    assert compute_quality(n_cycles=5, n_violations=0) == 1.0


def test_quality_zero_without_cycles():
    assert compute_quality(n_cycles=0, n_violations=3) == 0.0


def test_quality_decreases_with_violations():
    clean = compute_quality(n_cycles=4, n_violations=0)
    noisy = compute_quality(n_cycles=4, n_violations=4)
    assert noisy < clean
    assert 0.0 < noisy < 1.0


def test_quality_tempo_penalty_applies():
    a = compute_quality(n_cycles=4, n_violations=0, tempo_penalty=0.0)
    b = compute_quality(n_cycles=4, n_violations=0, tempo_penalty=2.0)
    assert b < a


def test_feedback_prioritises_sequential_over_geometric():
    violations = [
        {"type": GEOMETRIC, "constraint": "elbow", "deviation": 5, "feedback": "geo"},
        {"type": SEQUENTIAL, "constraint": "order", "feedback": "seq"},
    ]
    out = interpret_violations(violations)
    assert out[0] == "seq"


def test_feedback_ranks_larger_deviation_first():
    violations = [
        {"type": GEOMETRIC, "constraint": "a", "deviation": 5, "feedback": "small"},
        {"type": GEOMETRIC, "constraint": "b", "deviation": 40, "feedback": "big"},
    ]
    out = interpret_violations(violations)
    assert out[0] == "big"


def test_feedback_dedups_and_limits_top_k():
    violations = [
        {"type": GEOMETRIC, "constraint": "a", "deviation": 5, "feedback": "one"},
        {"type": GEOMETRIC, "constraint": "a", "deviation": 5, "feedback": "one"},
        {"type": GLOBAL, "constraint": "b", "feedback": "two"},
        {"type": DYNAMIC, "constraint": "c", "feedback": "three"},
        {"type": SEQUENTIAL, "constraint": "d", "feedback": "four"},
    ]
    out = interpret_violations(violations, top_k=2)
    assert len(out) == 2
    assert out[0] == "four"  # SEQUENTIAL highest priority
