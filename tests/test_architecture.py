"""Tests for the architecture.yaml components.

Covers the new data models, math utilities, AttentionCone architecture
methods, AttentionManager, NPCAttentionAgent, MultiHeadAttention, and
bridge protocol message types.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from vrspin import (
    AttentionCone,
    AttentionManager,
    AttentionResult,
    AttentionResultsMessage,
    MultiHeadAttention,
    NPCAttentionAgent,
    SceneEntity,
    UpdateSceneEntities,
    UpdateUserPose,
    direction_to_quaternion,
    forward_vector_from_quaternion,
    quaternion_distance,
    slerp,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _y_rot(deg: float) -> np.ndarray:
    return R.from_euler("y", deg, degrees=True).as_quat()


IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def _make_entity(name: str, orientation=IDENTITY, entity_type: str = "object"):
    return SceneEntity(
        id=name,
        position=[0.0, 0.0, 0.0],
        orientation=orientation,
        entity_type=entity_type,
    )


# ===========================================================================
# SceneEntity
# ===========================================================================


class TestSceneEntity:
    def test_construction(self):
        e = SceneEntity("obj1", [1, 0, 0], [0, 0, 0, 2], "object")
        assert e.id == "obj1"
        assert np.isclose(np.linalg.norm(e.orientation), 1.0)

    def test_metadata_defaults_to_empty(self):
        e = _make_entity("e1")
        assert e.metadata == {}

    def test_metadata_roundtrip(self):
        e = SceneEntity("e1", [0, 0, 0], IDENTITY, "panel", metadata={"page": 1})
        assert e.metadata["page"] == 1


# ===========================================================================
# AttentionResult
# ===========================================================================


class TestAttentionResult:
    def test_construction(self):
        r = AttentionResult("e1", True, 0.5, 0.7)
        assert r.entity_id == "e1"
        assert r.in_attention is True
        assert np.isclose(r.angular_distance, 0.5)
        assert np.isclose(r.weight, 0.7)

    def test_default_weight_is_zero(self):
        r = AttentionResult("e1", False, 1.0)
        assert r.weight == 0.0


# ===========================================================================
# Math Utilities
# ===========================================================================


class TestQuaternionDistance:
    def test_identity_distance_is_zero(self):
        assert np.isclose(quaternion_distance(IDENTITY, IDENTITY), 0.0, atol=1e-6)

    def test_90_degrees(self):
        d = quaternion_distance(IDENTITY, _y_rot(90.0))
        assert np.isclose(d, np.deg2rad(90.0), atol=1e-5)

    def test_zero_quaternion_returns_pi(self):
        assert np.isclose(quaternion_distance([0, 0, 0, 0], IDENTITY), np.pi)


class TestForwardVector:
    def test_identity_points_z(self):
        fwd = forward_vector_from_quaternion(IDENTITY)
        assert np.allclose(fwd, [0, 0, 1], atol=1e-6)

    def test_90_yaw(self):
        fwd = forward_vector_from_quaternion(_y_rot(90.0))
        expected = R.from_euler("y", 90, degrees=True).apply([0, 0, 1])
        assert np.allclose(fwd, expected, atol=1e-5)

    def test_zero_quaternion_returns_z(self):
        fwd = forward_vector_from_quaternion([0, 0, 0, 0])
        assert np.allclose(fwd, [0, 0, 1], atol=1e-6)


class TestDirectionToQuaternion:
    def test_forward_z_gives_identity(self):
        q = direction_to_quaternion([0, 0, 1])
        fwd = R.from_quat(q).apply([0, 0, 1])
        assert np.allclose(fwd, [0, 0, 1], atol=1e-6)

    def test_roundtrip(self):
        d = np.array([1.0, 0.0, 0.0])
        q = direction_to_quaternion(d)
        fwd = R.from_quat(q).apply([0, 0, 1])
        assert np.allclose(fwd, d, atol=1e-5)

    def test_backward_z(self):
        q = direction_to_quaternion([0, 0, -1])
        fwd = R.from_quat(q).apply([0, 0, 1])
        assert np.allclose(fwd, [0, 0, -1], atol=1e-5)

    def test_zero_vector_gives_identity(self):
        q = direction_to_quaternion([0, 0, 0])
        assert np.allclose(q, IDENTITY)


class TestSlerp:
    def test_t0_returns_start(self):
        q = slerp(IDENTITY, _y_rot(90), 0.0)
        assert np.allclose(q, IDENTITY, atol=1e-6)

    def test_t1_returns_end(self):
        end = _y_rot(90)
        q = slerp(IDENTITY, end, 1.0)
        assert np.allclose(q, end / np.linalg.norm(end), atol=1e-5)

    def test_t_half_is_midpoint(self):
        q = slerp(IDENTITY, _y_rot(90), 0.5)
        fwd = R.from_quat(q).apply([0, 0, 1])
        expected = R.from_euler("y", 45, degrees=True).apply([0, 0, 1])
        assert np.allclose(fwd, expected, atol=1e-5)

    def test_zero_quaternion_returns_identity(self):
        q = slerp([0, 0, 0, 0], IDENTITY, 0.5)
        assert np.allclose(q, IDENTITY)


# ===========================================================================
# AttentionCone — architecture-spec methods
# ===========================================================================


class TestAttentionConeArchMethods:
    def test_contains_alias(self):
        cone = AttentionCone(IDENTITY, np.deg2rad(30))
        assert cone.contains(IDENTITY)
        assert not cone.contains(_y_rot(90))

    def test_compute_distance_alias(self):
        cone = AttentionCone(IDENTITY, np.deg2rad(45))
        d = cone.compute_distance(_y_rot(90))
        assert np.isclose(d, np.deg2rad(90), atol=1e-5)

    def test_query_entities(self):
        cone = AttentionCone(IDENTITY, np.deg2rad(45))
        entities = [_make_entity("close"), _make_entity("far", _y_rot(90))]
        results = cone.query_entities(entities)
        assert len(results) == 2
        assert results[0].entity_id == "close"
        assert results[0].in_attention is True
        assert results[1].entity_id == "far"
        assert results[1].in_attention is False

    def test_query_entities_empty_list(self):
        cone = AttentionCone(IDENTITY, np.deg2rad(45))
        assert cone.query_entities([]) == []


# ===========================================================================
# AttentionManager
# ===========================================================================


class TestAttentionManager:
    def test_register_and_update(self):
        mgr = AttentionManager(half_angle_rad=np.deg2rad(60))
        mgr.register_entity(_make_entity("obj1"))
        results = mgr.update(IDENTITY)
        assert len(results) == 1
        assert results[0].in_attention is True

    def test_entity_outside_cone(self):
        mgr = AttentionManager(half_angle_rad=np.deg2rad(30))
        mgr.register_entity(_make_entity("far", _y_rot(90)))
        results = mgr.update(IDENTITY)
        assert results[0].in_attention is False

    def test_get_attended_entities(self):
        mgr = AttentionManager(half_angle_rad=np.deg2rad(60))
        mgr.register_entity(_make_entity("close"))
        mgr.register_entity(_make_entity("far", _y_rot(90)))
        mgr.update(IDENTITY)
        attended = mgr.get_attended_entities()
        ids = [e.id for e in attended]
        assert "close" in ids
        assert "far" not in ids

    def test_get_attended_before_update_returns_empty(self):
        mgr = AttentionManager()
        assert mgr.get_attended_entities() == []

    def test_register_replaces_entity(self):
        mgr = AttentionManager()
        mgr.register_entity(_make_entity("obj", IDENTITY, "type_a"))
        mgr.register_entity(_make_entity("obj", _y_rot(90), "type_b"))
        assert mgr.registered_entities["obj"].entity_type == "type_b"


# ===========================================================================
# NPCAttentionAgent
# ===========================================================================


class TestNPCAttentionAgent:
    def test_construction(self):
        agent = NPCAttentionAgent(IDENTITY)
        assert np.isclose(np.linalg.norm(agent.orientation), 1.0)
        assert agent.awareness == {}

    def test_zero_orientation_raises(self):
        with pytest.raises(ValueError):
            NPCAttentionAgent([0, 0, 0, 0])

    def test_update_adds_awareness(self):
        agent = NPCAttentionAgent(IDENTITY, perception_half_angle=np.deg2rad(120))
        entity = _make_entity("player")
        agent.update(IDENTITY, [entity])
        assert agent.is_aware_of("player")

    def test_update_entity_outside_cone_not_aware(self):
        agent = NPCAttentionAgent(IDENTITY, perception_half_angle=np.deg2rad(30))
        entity = _make_entity("far_entity", _y_rot(90))
        agent.update(IDENTITY, [entity])
        assert not agent.is_aware_of("far_entity")

    def test_confidence_decays(self):
        agent = NPCAttentionAgent(IDENTITY, perception_half_angle=np.deg2rad(120),
                                  confidence_decay=0.5)
        entity = _make_entity("player")
        agent.update(IDENTITY, [entity])
        assert agent.is_aware_of("player")
        # Update with entity now outside cone
        far_entity = _make_entity("player", _y_rot(170))
        agent.update(IDENTITY, [far_entity])
        # Confidence decayed but still above threshold
        assert agent.is_aware_of("player")
        assert agent.awareness["player"].confidence < 1.0

    def test_confidence_fully_decays(self):
        agent = NPCAttentionAgent(IDENTITY, perception_half_angle=np.deg2rad(120),
                                  confidence_decay=0.001)
        entity = _make_entity("player")
        agent.update(IDENTITY, [entity])
        # Move entity out of cone and update many times
        far_entity = _make_entity("player", _y_rot(170))
        for _ in range(5):
            agent.update(IDENTITY, [far_entity])
        assert not agent.is_aware_of("player")

    def test_compute_rotation_to_target(self):
        agent = NPCAttentionAgent(IDENTITY)
        target = _y_rot(90)
        delta = agent.compute_rotation_to_target(target)
        composed = R.from_quat(agent.orientation) * R.from_quat(delta)
        assert np.allclose(
            composed.as_quat(),
            target / np.linalg.norm(target),
            atol=1e-5,
        )

    def test_compute_rotation_zero_target(self):
        agent = NPCAttentionAgent(IDENTITY)
        delta = agent.compute_rotation_to_target([0, 0, 0, 0])
        assert np.allclose(delta, IDENTITY)


# ===========================================================================
# MultiHeadAttention
# ===========================================================================


class TestMultiHeadAttention:
    def _make_mha(self):
        v = AttentionCone(IDENTITY, np.deg2rad(60), "visual")
        a = AttentionCone(IDENTITY, np.deg2rad(120), "audio")
        h = AttentionCone(IDENTITY, np.deg2rad(30), "haptic")
        return MultiHeadAttention(v, a, h)

    def test_update_returns_results(self):
        mha = self._make_mha()
        entities = [_make_entity("obj")]
        results = mha.update(entities)
        assert len(results) == 1
        assert results[0].in_attention is True
        assert results[0].weight > 0.0

    def test_entity_in_audio_only(self):
        mha = self._make_mha()
        # 80° — outside visual (60°) and haptic (30°), inside audio (120°)
        entities = [_make_entity("audio_obj", _y_rot(80))]
        results = mha.update(entities)
        assert results[0].in_attention is True
        # Weight should only reflect audio contribution
        assert results[0].weight > 0.0
        # Weight should be less than if also in visual
        full_results = mha.update([_make_entity("full", IDENTITY)])
        assert full_results[0].weight > results[0].weight

    def test_entity_outside_all_cones(self):
        mha = self._make_mha()
        entities = [_make_entity("far", _y_rot(170))]
        results = mha.update(entities)
        assert results[0].in_attention is False
        assert results[0].weight == 0.0

    def test_custom_weights(self):
        v = AttentionCone(IDENTITY, np.deg2rad(60), "visual")
        a = AttentionCone(IDENTITY, np.deg2rad(120), "audio")
        h = AttentionCone(IDENTITY, np.deg2rad(30), "haptic")
        mha = MultiHeadAttention(v, a, h, weights={"visual": 1.0, "audio": 0.0, "haptic": 0.0})
        entities = [_make_entity("obj")]
        results = mha.update(entities)
        # Only visual contributes
        assert results[0].weight > 0.0

    def test_merge_results_directly(self):
        mha = self._make_mha()
        r_vis = [AttentionResult("e1", True, 0.1)]
        r_aud = [AttentionResult("e1", True, 0.5)]
        r_hap = [AttentionResult("e1", False, 2.0)]
        merged = mha.merge_results({"visual": r_vis, "audio": r_aud, "haptic": r_hap})
        assert len(merged) == 1
        assert merged[0].in_attention is True
        assert merged[0].weight > 0.0

    def test_default_weights_match_spec(self):
        mha = self._make_mha()
        assert np.isclose(mha.weights["visual"], 0.7)
        assert np.isclose(mha.weights["audio"], 0.2)
        assert np.isclose(mha.weights["haptic"], 0.1)


# ===========================================================================
# Bridge Protocol
# ===========================================================================


class TestBridge:
    def test_update_user_pose_roundtrip(self):
        msg = UpdateUserPose([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0])
        d = msg.to_dict()
        assert d["type"] == "update_user_pose"
        restored = UpdateUserPose.from_dict(d)
        assert restored.position == msg.position
        assert restored.orientation == msg.orientation

    def test_update_scene_entities_roundtrip(self):
        entity = SceneEntity("e1", [0, 0, 0], IDENTITY, "object", {"key": "val"})
        msg = UpdateSceneEntities([entity])
        d = msg.to_dict()
        assert d["type"] == "update_scene_entities"
        restored = UpdateSceneEntities.from_dict(d)
        assert len(restored.entities) == 1
        assert restored.entities[0].id == "e1"
        assert restored.entities[0].metadata["key"] == "val"

    def test_attention_results_message_roundtrip(self):
        result = AttentionResult("e1", True, 0.5, 0.8)
        msg = AttentionResultsMessage([result])
        d = msg.to_dict()
        assert d["type"] == "attention_results"
        restored = AttentionResultsMessage.from_dict(d)
        assert len(restored.results) == 1
        assert restored.results[0].entity_id == "e1"
        assert restored.results[0].in_attention is True
        assert np.isclose(restored.results[0].angular_distance, 0.5)
        assert np.isclose(restored.results[0].weight, 0.8)
