"""Tests for vrspin.scene — SceneEntity, AttentionResult, AttentionManager."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from vrspin.cone import AttentionCone
from vrspin.scene import AttentionManager, AttentionResult, SceneEntity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def _y_rot(deg: float) -> np.ndarray:
    """Quaternion for a pure Y-axis rotation of *deg* degrees."""
    return R.from_euler("y", deg, degrees=True).as_quat()


# ---------------------------------------------------------------------------
# SceneEntity
# ---------------------------------------------------------------------------


class TestSceneEntity:
    def test_construction_basic(self):
        ent = SceneEntity("fountain", IDENTITY, [5, 0, 3], "object")
        assert ent.name == "fountain"
        assert ent.entity_type == "object"
        np.testing.assert_allclose(ent.position, [5, 0, 3])

    def test_inherits_node(self):
        from spinstep import Node

        ent = SceneEntity("x", IDENTITY)
        assert isinstance(ent, Node)

    def test_default_position_is_origin(self):
        ent = SceneEntity("x", IDENTITY)
        np.testing.assert_allclose(ent.position, [0, 0, 0])

    def test_direction_quaternion(self):
        q = _y_rot(30)
        ent = SceneEntity("a", q)
        np.testing.assert_allclose(ent.direction_quaternion, ent.orientation)

    def test_distance_to(self):
        a = SceneEntity("a", IDENTITY, [0, 0, 0])
        b = SceneEntity("b", IDENTITY, [3, 4, 0])
        assert pytest.approx(a.distance_to(b), abs=1e-6) == 5.0

    def test_metadata(self):
        ent = SceneEntity("panel", IDENTITY, metadata={"content": "Hello"})
        assert ent.metadata["content"] == "Hello"

    def test_repr(self):
        ent = SceneEntity("f", IDENTITY, [1, 2, 3], "npc")
        r = repr(ent)
        assert "f" in r
        assert "npc" in r


# ---------------------------------------------------------------------------
# AttentionResult
# ---------------------------------------------------------------------------


class TestAttentionResult:
    def test_default_construction(self):
        ar = AttentionResult()
        assert ar.attended == []
        assert ar.unattended == []

    def test_with_data(self):
        ent = SceneEntity("a", IDENTITY)
        ar = AttentionResult(attended=[(ent, 0.9)], unattended=[])
        assert len(ar.attended) == 1
        assert ar.attended[0][1] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# AttentionManager
# ---------------------------------------------------------------------------


class TestAttentionManager:
    def test_register_and_list(self):
        mgr = AttentionManager()
        e = SceneEntity("a", IDENTITY)
        mgr.register_entity(e)
        assert len(mgr.entities) == 1
        assert mgr.entities[0].name == "a"

    def test_unregister(self):
        e = SceneEntity("a", IDENTITY)
        mgr = AttentionManager([e])
        mgr.unregister_entity("a")
        assert len(mgr.entities) == 0

    def test_unregister_unknown_raises(self):
        mgr = AttentionManager()
        with pytest.raises(KeyError):
            mgr.unregister_entity("missing")

    def test_update_finds_aligned_entity(self):
        ent = SceneEntity("fountain", IDENTITY, [5, 0, 3], "object")
        mgr = AttentionManager([ent])
        result = mgr.update(IDENTITY, cone_half_angle=np.radians(45))
        assert len(result.attended) == 1
        assert result.attended[0][0].name == "fountain"
        assert result.attended[0][1] > 0

    def test_update_excludes_far_entity(self):
        ent = SceneEntity("far", _y_rot(90), [5, 0, 3], "object")
        mgr = AttentionManager([ent])
        result = mgr.update(IDENTITY, cone_half_angle=np.radians(10))
        assert len(result.attended) == 0
        assert len(result.unattended) == 1

    def test_update_sorts_by_strength(self):
        close = SceneEntity("close", _y_rot(5))
        far = SceneEntity("far", _y_rot(30))
        mgr = AttentionManager([far, close])
        result = mgr.update(IDENTITY, cone_half_angle=np.radians(45))
        # 'close' should have higher attenuation → first
        assert result.attended[0][0].name == "close"

    def test_update_with_no_entities(self):
        mgr = AttentionManager()
        result = mgr.update(IDENTITY, cone_half_angle=0.5)
        assert result.attended == []
        assert result.unattended == []

    def test_init_with_entities_list(self):
        entities = [
            SceneEntity("a", IDENTITY),
            SceneEntity("b", _y_rot(45)),
        ]
        mgr = AttentionManager(entities)
        assert len(mgr.entities) == 2

    def test_get_attended_entities_after_update(self):
        close = SceneEntity("close", _y_rot(5))
        far = SceneEntity("far", _y_rot(90))
        mgr = AttentionManager([close, far])
        mgr.update(IDENTITY, cone_half_angle=np.radians(45))
        attended = mgr.get_attended_entities()
        assert len(attended) == 1
        assert attended[0].name == "close"

    def test_get_attended_entities_before_update(self):
        mgr = AttentionManager()
        assert mgr.get_attended_entities() == []


# ---------------------------------------------------------------------------
# Multi-observer AttentionManager
# ---------------------------------------------------------------------------


class TestMultiObserver:
    """Tests for AttentionManager.update_observers and Observer protocol."""

    def test_update_observers_two_observers(self):
        """Two observers with different orientations get independent results."""

        class _SimpleObserver:
            def __init__(self, name, orientation):
                self.name = name
                self._orientation = np.asarray(orientation, dtype=float)
                self._cones = {
                    "visual": AttentionCone(
                        self._orientation, half_angle=np.radians(45), falloff="linear",
                    ),
                }

            @property
            def orientation(self):
                return self._orientation

            @property
            def attention_cones(self):
                return self._cones

        obs1 = _SimpleObserver("obs1", IDENTITY)
        obs2 = _SimpleObserver("obs2", _y_rot(90))

        e1 = SceneEntity("front", IDENTITY)
        e2 = SceneEntity("left", _y_rot(90))
        e3 = SceneEntity("behind", _y_rot(180))

        mgr = AttentionManager([e1, e2, e3])
        results = mgr.update_observers(
            [obs1, obs2], cone_half_angle=np.radians(45),
        )

        assert "obs1" in results
        assert "obs2" in results
        # obs1 faces forward → should attend 'front', not 'left' or 'behind'
        obs1_names = [ent.name for ent, _ in results["obs1"].attended]
        assert "front" in obs1_names
        assert "behind" not in obs1_names
        # obs2 faces 90° → should attend 'left'
        obs2_names = [ent.name for ent, _ in results["obs2"].attended]
        assert "left" in obs2_names
        assert "behind" not in obs2_names

    def test_update_observers_backward_compatible(self):
        """mgr.update() still works with a single user_quat."""
        ent = SceneEntity("a", _y_rot(5))
        mgr = AttentionManager([ent])
        result = mgr.update(IDENTITY, cone_half_angle=np.radians(45))
        assert isinstance(result, AttentionResult)
        assert len(result.attended) == 1
        assert result.attended[0][0].name == "a"

    def test_observer_protocol_check(self):
        """An object with orientation and attention_cones satisfies Observer."""
        from vrspin.scene import Observer

        class _MyObserver:
            @property
            def orientation(self):
                return IDENTITY

            @property
            def attention_cones(self):
                return {"visual": AttentionCone(IDENTITY, half_angle=0.5)}

        obs = _MyObserver()
        assert isinstance(obs, Observer)
