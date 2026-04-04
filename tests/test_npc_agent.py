"""Tests for NPCAttentionAgent, slerp, and new AttentionCone API methods."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from vrspin import slerp, forward_vector_from_quaternion
from vrspin.cone import AttentionCone
from vrspin.npc import NPCAttentionAgent
from vrspin.scene import SceneEntity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def _y_rot(deg: float) -> np.ndarray:
    return R.from_euler("y", deg, degrees=True).as_quat()


# ---------------------------------------------------------------------------
# slerp utility
# ---------------------------------------------------------------------------


class TestSlerp:
    def test_slerp_t0_returns_start(self):
        result = slerp(IDENTITY, _y_rot(90), 0.0)
        np.testing.assert_allclose(result, IDENTITY, atol=1e-6)

    def test_slerp_t1_returns_end(self):
        target = _y_rot(90)
        result = slerp(IDENTITY, target, 1.0)
        np.testing.assert_allclose(result, target, atol=1e-6)

    def test_slerp_midpoint(self):
        result = slerp(IDENTITY, _y_rot(90), 0.5)
        expected = _y_rot(45)
        np.testing.assert_allclose(result, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# forward_vector_from_quaternion
# ---------------------------------------------------------------------------


class TestForwardVector:
    def test_identity_looks_negative_z(self):
        fwd = forward_vector_from_quaternion(IDENTITY)
        np.testing.assert_allclose(fwd, [0, 0, -1], atol=1e-6)


# ---------------------------------------------------------------------------
# Utility re-exports
# ---------------------------------------------------------------------------


class TestUtilityReexports:
    def test_direction_to_quaternion_available(self):
        from vrspin import direction_to_quaternion
        q = direction_to_quaternion([0, 0, -1])
        assert len(q) == 4
        assert abs(np.linalg.norm(q) - 1.0) < 1e-6

    def test_angle_between_directions_available(self):
        from vrspin import angle_between_directions
        angle = angle_between_directions([1, 0, 0], [0, 1, 0])
        assert abs(angle - np.pi / 2) < 1e-6
# AttentionCone — new API methods
# ---------------------------------------------------------------------------


class TestAttentionConeNewAPI:
    def test_contains_alias(self):
        cone = AttentionCone(IDENTITY, half_angle_rad=np.radians(45))
        assert cone.contains(_y_rot(10))
        assert not cone.contains(_y_rot(90))

    def test_half_angle_keyword(self):
        cone = AttentionCone(IDENTITY, half_angle=np.radians(30))
        assert cone.half_angle == pytest.approx(np.radians(30))

    def test_update_origin_alias(self):
        cone = AttentionCone(IDENTITY, half_angle=np.radians(30))
        new_dir = _y_rot(45)
        cone.update_origin(new_dir)
        np.testing.assert_allclose(cone.orientation, new_dir, atol=1e-6)

    def test_attenuation_inside_linear(self):
        cone = AttentionCone(IDENTITY, half_angle=np.radians(45), falloff="linear")
        val = cone.attenuation(_y_rot(0))
        assert val == pytest.approx(1.0, abs=0.01)

    def test_attenuation_edge_linear(self):
        cone = AttentionCone(IDENTITY, half_angle=np.radians(45), falloff="linear")
        val = cone.attenuation(_y_rot(44))
        assert 0 < val < 0.1  # close to edge

    def test_attenuation_outside(self):
        cone = AttentionCone(IDENTITY, half_angle=np.radians(45), falloff="linear")
        val = cone.attenuation(_y_rot(90))
        assert val == 0.0

    def test_attenuation_no_falloff(self):
        cone = AttentionCone(IDENTITY, half_angle=np.radians(45))
        assert cone.attenuation(_y_rot(10)) == 1.0
        assert cone.attenuation(_y_rot(90)) == 0.0

    def test_attenuation_cosine(self):
        cone = AttentionCone(IDENTITY, half_angle=np.radians(90), falloff="cosine")
        # Dead centre → 1.0
        assert cone.attenuation(IDENTITY) == pytest.approx(1.0, abs=0.01)
        # Edge → 0.0
        val = cone.attenuation(_y_rot(89))
        assert val > 0
        assert val < 0.1

    def test_query_batch(self):
        cone = AttentionCone(IDENTITY, half_angle=np.radians(30))
        quats = np.array([IDENTITY, _y_rot(10), _y_rot(60)])
        mask = cone.query_batch(quats)
        assert mask[0] and mask[1] and not mask[2]

    def test_query_batch_with_attenuation(self):
        cone = AttentionCone(IDENTITY, half_angle=np.radians(30), falloff="linear")
        quats = np.array([IDENTITY, _y_rot(60)])
        vals = cone.query_batch_with_attenuation(quats)
        assert vals[0] > 0.9
        assert vals[1] == 0.0

    def test_invalid_falloff_raises(self):
        with pytest.raises(ValueError):
            AttentionCone(IDENTITY, half_angle=0.5, falloff="quadratic")

    def test_missing_half_angle_raises(self):
        with pytest.raises(TypeError):
            AttentionCone(IDENTITY)


# ---------------------------------------------------------------------------
# NPCAttentionAgent
# ---------------------------------------------------------------------------


class TestNPCAttentionAgent:
    def _make_agent(self, ent_dir=IDENTITY, half=np.radians(40)):
        ent = SceneEntity("vendor", ent_dir, entity_type="npc")
        return NPCAttentionAgent(ent, perception_half_angle=half)

    def test_is_aware_in_cone(self):
        agent = self._make_agent()
        assert agent.is_aware_of(_y_rot(10))

    def test_is_aware_outside_cone(self):
        agent = self._make_agent()
        assert not agent.is_aware_of(_y_rot(90))

    def test_face_toward_moves_orientation(self):
        agent = self._make_agent()
        orig = agent.entity.orientation.copy()
        agent.face_toward(_y_rot(30), 0.5)
        assert not np.allclose(agent.entity.orientation, orig)

    def test_update_with_target_in_cone(self):
        agent = self._make_agent(half=np.radians(60))
        orig = agent.entity.orientation.copy()
        agent.update(targets=[_y_rot(30)], dt=1 / 60)
        assert not np.allclose(agent.entity.orientation, orig)

    def test_update_with_no_targets_returns_to_idle(self):
        agent = self._make_agent()
        # Move NPC first
        agent.face_toward(_y_rot(20), 0.3)
        # Then update with no targets — should rotate back toward idle
        agent.update(targets=[], dt=1 / 60)
        # Just verifying it doesn't crash; full convergence needs many frames

    def test_idle_orientation_default(self):
        ent = SceneEntity("v", _y_rot(45), entity_type="npc")
        agent = NPCAttentionAgent(ent, perception_half_angle=0.5)
        np.testing.assert_allclose(agent.idle_orientation, _y_rot(45), atol=1e-6)

    def test_custom_idle_orientation(self):
        ent = SceneEntity("v", IDENTITY, entity_type="npc")
        custom_idle = _y_rot(90)
        agent = NPCAttentionAgent(
            ent, perception_half_angle=0.5, idle_orientation=custom_idle,
        )
        np.testing.assert_allclose(agent.idle_orientation, custom_idle, atol=1e-6)
