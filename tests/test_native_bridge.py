"""Tests for the VRSpin native C bridge.

Validates that the native (ctypes) implementation produces the same
results as the pure-Python :class:`~vrspin.cone.AttentionCone`.

The tests require the native library to be compiled first::

    cd vrspin/native && make
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skip the entire module if the native library is not available.
# ---------------------------------------------------------------------------

_NATIVE_DIR = Path(__file__).resolve().parent.parent / "vrspin" / "native"
_SO_FILE = _NATIVE_DIR / "libvrspin_native.so"
_DYLIB_FILE = _NATIVE_DIR / "libvrspin_native.dylib"
_DLL_FILE = _NATIVE_DIR / "vrspin_native.dll"

_LIB_AVAILABLE = _SO_FILE.exists() or _DYLIB_FILE.exists() or _DLL_FILE.exists()

pytestmark = pytest.mark.skipif(
    not _LIB_AVAILABLE,
    reason="Native library not compiled — run 'cd vrspin/native && make' first",
)

# Only import after the skip check so collection doesn't fail when lib is absent.
if _LIB_AVAILABLE:
    from vrspin.native_bridge import (
        NativeAttentionCone,
        native_forward_vector,
        native_process_frame,
        native_quat_distance,
        native_slerp,
    )

from vrspin.cone import AttentionCone
from vrspin.utils import slerp as py_slerp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])
RIGHT_45 = np.array([0.0, np.sin(np.radians(22.5)), 0.0, np.cos(np.radians(22.5))])
RIGHT_90 = np.array([0.0, np.sin(np.radians(45)), 0.0, np.cos(np.radians(45))])
HALF_ANGLE_45 = np.radians(45)
HALF_ANGLE_90 = np.radians(90)


def _close(a: float, b: float, atol: float = 1e-6) -> bool:
    return abs(a - b) < atol


# ===========================================================================
# Quaternion utility tests
# ===========================================================================


class TestNativeQuatDistance:
    """vrspin_quat_distance matches Python AttentionCone.angular_distance_to."""

    def test_identical_quaternions(self) -> None:
        assert _close(native_quat_distance(IDENTITY, IDENTITY), 0.0)

    def test_right_angle(self) -> None:
        dist = native_quat_distance(IDENTITY, RIGHT_90)
        expected = AttentionCone(IDENTITY, half_angle=1.0).angular_distance_to(RIGHT_90)
        assert _close(dist, expected)

    def test_opposite_quaternions(self) -> None:
        opposite = np.array([0.0, 1.0, 0.0, 0.0])  # 180° around Y
        dist = native_quat_distance(IDENTITY, opposite)
        assert _close(dist, np.pi, atol=1e-4)

    def test_small_angle(self) -> None:
        small = np.array([0.0, np.sin(0.01), 0.0, np.cos(0.01)])
        dist = native_quat_distance(IDENTITY, small)
        assert _close(dist, 0.02, atol=1e-3)


class TestNativeForwardVector:
    """vrspin_forward_vector matches AttentionCone.get_forward_vector."""

    def test_identity_gives_neg_z(self) -> None:
        fwd = native_forward_vector(IDENTITY)
        np.testing.assert_allclose(fwd, [0.0, 0.0, -1.0], atol=1e-10)

    def test_matches_python(self) -> None:
        cone = AttentionCone(RIGHT_45, half_angle=1.0)
        py_fwd = cone.get_forward_vector()
        native_fwd = native_forward_vector(RIGHT_45)
        np.testing.assert_allclose(native_fwd, py_fwd, atol=1e-10)

    def test_90_deg_rotation(self) -> None:
        fwd = native_forward_vector(RIGHT_90)
        # 90° around Y should rotate -Z to -X
        np.testing.assert_allclose(fwd, [-1.0, 0.0, 0.0], atol=1e-10)


class TestNativeSlerp:
    """vrspin_slerp matches vrspin.utils.slerp."""

    def test_t0_returns_start(self) -> None:
        result = native_slerp(IDENTITY, RIGHT_90, 0.0)
        np.testing.assert_allclose(result, IDENTITY, atol=1e-10)

    def test_t1_returns_end(self) -> None:
        result = native_slerp(IDENTITY, RIGHT_90, 1.0)
        np.testing.assert_allclose(np.abs(result), np.abs(RIGHT_90), atol=1e-10)

    def test_midpoint_matches_python(self) -> None:
        native_mid = native_slerp(IDENTITY, RIGHT_90, 0.5)
        py_mid = py_slerp(IDENTITY, RIGHT_90, 0.5)
        np.testing.assert_allclose(native_mid, py_mid, atol=1e-6)

    def test_quarter_point(self) -> None:
        native_q = native_slerp(IDENTITY, RIGHT_90, 0.25)
        py_q = py_slerp(IDENTITY, RIGHT_90, 0.25)
        np.testing.assert_allclose(native_q, py_q, atol=1e-6)


# ===========================================================================
# Attention cone tests
# ===========================================================================


class TestNativeAttentionConeContains:
    """NativeAttentionCone.contains matches AttentionCone.contains."""

    def test_inside_cone(self) -> None:
        native = NativeAttentionCone(IDENTITY, half_angle=HALF_ANGLE_90)
        py = AttentionCone(IDENTITY, half_angle=HALF_ANGLE_90)
        assert native.contains(RIGHT_45) == py.contains(RIGHT_45)
        assert native.contains(RIGHT_45) is True

    def test_outside_cone(self) -> None:
        native = NativeAttentionCone(IDENTITY, half_angle=HALF_ANGLE_45)
        opposite = np.array([0.0, 1.0, 0.0, 0.0])
        assert native.contains(opposite) is False

    def test_edge_case_exact_boundary(self) -> None:
        # At the exact boundary the result depends on floating-point
        # rounding in the two implementations.  We only assert that
        # *both* implementations agree on targets clearly inside or
        # clearly outside.
        angle = 0.5
        # Slightly inside (99% of half_angle)
        inside = np.array([0.0, np.sin(angle * 0.99 / 2), 0.0, np.cos(angle * 0.99 / 2)])
        # Slightly outside (101% of half_angle)
        outside = np.array([0.0, np.sin(angle * 1.01 / 2), 0.0, np.cos(angle * 1.01 / 2)])
        native = NativeAttentionCone(IDENTITY, half_angle=angle)
        py = AttentionCone(IDENTITY, half_angle=angle)
        assert native.contains(inside) is True
        assert py.contains(inside) is True
        assert native.contains(outside) is False
        assert py.contains(outside) is False

    def test_is_in_cone_alias(self) -> None:
        native = NativeAttentionCone(IDENTITY, half_angle=HALF_ANGLE_90)
        assert native.is_in_cone(RIGHT_45) == native.contains(RIGHT_45)


class TestNativeAttentionConeAttenuation:
    """NativeAttentionCone.attenuation matches AttentionCone.attenuation."""

    @pytest.mark.parametrize("falloff", [None, "linear", "cosine"])
    def test_centre_gives_max(self, falloff: str) -> None:
        native = NativeAttentionCone(IDENTITY, half_angle=1.0, falloff=falloff)
        py = AttentionCone(IDENTITY, half_angle=1.0, falloff=falloff)
        assert _close(native.attenuation(IDENTITY), py.attenuation(IDENTITY))

    @pytest.mark.parametrize("falloff", [None, "linear", "cosine"])
    def test_outside_gives_zero(self, falloff: str) -> None:
        native = NativeAttentionCone(IDENTITY, half_angle=0.3, falloff=falloff)
        opposite = np.array([0.0, 1.0, 0.0, 0.0])
        assert _close(native.attenuation(opposite), 0.0)

    @pytest.mark.parametrize("falloff", ["linear", "cosine"])
    def test_midway_matches_python(self, falloff: str) -> None:
        native = NativeAttentionCone(IDENTITY, half_angle=HALF_ANGLE_90, falloff=falloff)
        py = AttentionCone(IDENTITY, half_angle=HALF_ANGLE_90, falloff=falloff)
        assert _close(native.attenuation(RIGHT_45), py.attenuation(RIGHT_45), atol=1e-4)


class TestNativeAttentionConeUpdateOrigin:
    """NativeAttentionCone.update_origin changes the cone direction."""

    def test_update_changes_containment(self) -> None:
        native = NativeAttentionCone(IDENTITY, half_angle=0.3)
        # RIGHT_90 is far outside a narrow cone centered at identity
        assert native.contains(RIGHT_90) is False
        # After re-centering the cone on RIGHT_90, it should contain RIGHT_90
        native.update_origin(RIGHT_90)
        assert native.contains(RIGHT_90) is True

    def test_update_orientation_alias(self) -> None:
        native = NativeAttentionCone(IDENTITY, half_angle=0.3)
        native.update_orientation(RIGHT_90)
        assert native.contains(RIGHT_90) is True


class TestNativeAttentionConeBatch:
    """Batch queries match single-item queries."""

    def test_query_batch_matches_individual(self) -> None:
        quats = np.array([IDENTITY, RIGHT_45, RIGHT_90])
        native = NativeAttentionCone(IDENTITY, half_angle=HALF_ANGLE_90)
        batch = native.query_batch(quats)
        individual = [native.contains(q) for q in quats]
        np.testing.assert_array_equal(batch, individual)

    def test_query_batch_attenuation_matches_individual(self) -> None:
        quats = np.array([IDENTITY, RIGHT_45, RIGHT_90])
        native = NativeAttentionCone(IDENTITY, half_angle=HALF_ANGLE_90, falloff="linear")
        batch = native.query_batch_with_attenuation(quats)
        individual = [native.attenuation(q) for q in quats]
        np.testing.assert_allclose(batch, individual, atol=1e-10)


class TestNativeAttentionConeMatchesPython:
    """Cross-validate native cone against the pure-Python implementation."""

    @pytest.mark.parametrize("falloff", [None, "linear", "cosine"])
    def test_batch_attenuation_matches_python(self, falloff: str) -> None:
        quats = np.array([
            IDENTITY,
            RIGHT_45,
            RIGHT_90,
            [0.0, 1.0, 0.0, 0.0],  # 180°
            [0.1, 0.0, 0.0, 0.995],
        ])
        native = NativeAttentionCone(IDENTITY, half_angle=HALF_ANGLE_90, falloff=falloff)
        py = AttentionCone(IDENTITY, half_angle=HALF_ANGLE_90, falloff=falloff)

        native_att = native.query_batch_with_attenuation(quats)
        py_att = py.query_batch_with_attenuation(quats)
        np.testing.assert_allclose(native_att, py_att, atol=2e-4)


# ===========================================================================
# Frame processing test
# ===========================================================================


class TestNativeProcessFrame:
    """vrspin_process_frame produces correct visual and audio results."""

    def test_frame_processing(self) -> None:
        entities = np.array([
            IDENTITY,           # dead centre
            RIGHT_45,           # 45° off
            RIGHT_90,           # 90° off
            [0.0, 1.0, 0.0, 0.0],  # 180° off
        ])
        vis_str, aud_gain, vis_att, count = native_process_frame(
            IDENTITY, entities,
            visual_half_angle=HALF_ANGLE_90,
            audio_half_angle=np.radians(120),
        )
        # Identity entity is dead centre — max visual
        assert vis_str[0] > 0.9
        assert vis_att[0] is np.True_
        # 180° entity should not be attended
        assert vis_str[3] == 0.0
        assert vis_att[3] is np.False_
        # Audio cone is wider — should reach more entities
        assert aud_gain[0] > 0.0
        assert count >= 2  # at least identity + RIGHT_45

    def test_frame_count_matches_sum(self) -> None:
        entities = np.array([IDENTITY, RIGHT_45, RIGHT_90])
        _, _, vis_att, count = native_process_frame(
            IDENTITY, entities,
            visual_half_angle=HALF_ANGLE_45,
            audio_half_angle=HALF_ANGLE_90,
        )
        assert count == int(np.sum(vis_att))


# ===========================================================================
# Error handling
# ===========================================================================


class TestNativeAttentionConeErrors:
    """Error handling in the native bridge."""

    def test_invalid_falloff_raises(self) -> None:
        with pytest.raises(ValueError, match="falloff"):
            NativeAttentionCone(IDENTITY, half_angle=1.0, falloff="invalid")

    def test_repr(self) -> None:
        cone = NativeAttentionCone(IDENTITY, half_angle=0.5, falloff="linear")
        r = repr(cone)
        assert "NativeAttentionCone" in r
        assert "0.5000" in r
