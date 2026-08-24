"""Unit tests for geometric angle primitives."""

import math

import numpy as np
import pytest

from rehubai.angles import (
    angle_deviation,
    calculate_angle,
    confidence_ok,
    vector_angle,
    within_tolerance,
)


def test_straight_line_is_180():
    assert calculate_angle([0, 0], [1, 0], [2, 0]) == pytest.approx(180.0)


def test_right_angle_is_90():
    assert calculate_angle([0, 1], [0, 0], [1, 0]) == pytest.approx(90.0)


def test_zero_angle_when_rays_coincide():
    assert calculate_angle([1, 0], [0, 0], [1, 0]) == pytest.approx(0.0)


def test_arbitrary_angle_150():
    p0 = [math.cos(math.radians(150)), math.sin(math.radians(150))]
    assert calculate_angle(p0, [0, 0], [1, 0]) == pytest.approx(150.0, abs=1e-6)


def test_degenerate_vector_returns_zero():
    # point2 == point1 => zero-length bone => defined as 0.
    assert calculate_angle([0, 0], [0, 0], [1, 0]) == 0.0


def test_works_in_3d():
    assert calculate_angle([1, 0, 0], [0, 0, 0], [0, 0, 1]) == pytest.approx(90.0)


def test_vector_angle_matches_calculate_angle():
    # angle between shoulder->elbow and shoulder->hip
    assert vector_angle([0, 0], [0, 1], [1, 0]) == pytest.approx(90.0)


def test_within_tolerance_and_deviation():
    assert within_tolerance(175, 180, 10) is True
    assert within_tolerance(160, 180, 10) is False
    assert angle_deviation(190, 180) == pytest.approx(10.0)


def test_confidence_gate():
    conf = [0.9, 0.2, 0.8]
    assert confidence_ok(conf, [0, 2], 0.5) is True
    assert confidence_ok(conf, [0, 1], 0.5) is False
    assert confidence_ok(None, [0, 1], 0.5) is True  # no scores => pass
