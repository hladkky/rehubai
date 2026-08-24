"""Unit tests for the semantic joint abstraction."""

import pytest

from rehubai import joints


def test_resolve_blazepose():
    assert joints.resolve("blazepose33", "left_elbow") == 13
    assert joints.resolve("blazepose33", "right_ankle") == 28


def test_same_semantic_name_maps_across_schemes():
    for scheme in ("blazepose33", "kinect25", "coco17"):
        idx = joints.resolve(scheme, "left_shoulder")
        assert isinstance(idx, int)


def test_resolve_many_preserves_order():
    assert joints.resolve_many("coco17", ["left_hip", "left_knee", "left_ankle"]) == [11, 13, 15]


def test_supports():
    assert joints.supports("coco17", ["left_elbow", "right_knee"]) is True
    # COCO has no explicit neck joint.
    assert joints.supports("coco17", ["neck"]) is False


def test_unknown_scheme_raises():
    with pytest.raises(KeyError):
        joints.resolve("does_not_exist", "left_elbow")


def test_unknown_joint_raises():
    with pytest.raises(KeyError):
        joints.resolve("coco17", "spine_base")


def test_register_scheme_rejects_unknown_semantic():
    with pytest.raises(ValueError):
        joints.register_scheme("bad", {"not_a_real_joint": 0})


def test_register_and_use_custom_scheme():
    joints.register_scheme("toy3", {"left_elbow": 0, "left_shoulder": 1, "left_wrist": 2})
    assert joints.resolve("toy3", "left_wrist") == 2
    assert "toy3" in joints.available_schemes()
