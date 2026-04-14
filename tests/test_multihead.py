"""Tests for vrspin.multihead — MultiHeadAttention."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from vrspin.cone import AttentionCone
from vrspin.multihead import MultiHeadAttention
from vrspin.scene import SceneEntity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def _y_rot(deg: float) -> np.ndarray:
    return R.from_euler("y", deg, degrees=True).as_quat()


# ---------------------------------------------------------------------------
# MultiHeadAttention
# ---------------------------------------------------------------------------


class TestMultiHeadAttention:
    def _make_heads(self, origin=IDENTITY):
        return {
            "visual": AttentionCone(origin, half_angle=np.radians(30), falloff="linear"),
            "audio": AttentionCone(origin, half_angle=np.radians(90), falloff="cosine"),
            "haptic": AttentionCone(origin, half_angle=np.radians(15), falloff="linear"),
        }

    def test_construction(self):
        multi = MultiHeadAttention(self._make_heads())
        assert set(multi.heads.keys()) == {"visual", "audio", "haptic"}

    def test_update_returns_per_head(self):
        multi = MultiHeadAttention(self._make_heads())
        ent = SceneEntity("close", _y_rot(10))
        results = multi.update(IDENTITY, [ent])
        assert "visual" in results
        assert "audio" in results
        assert "haptic" in results

    def test_wider_cone_captures_more(self):
        multi = MultiHeadAttention(self._make_heads())
        close = SceneEntity("close", _y_rot(10))
        mid = SceneEntity("mid", _y_rot(50))
        far = SceneEntity("far", _y_rot(100))
        results = multi.update(IDENTITY, [close, mid, far])
        # visual (30°) should only contain 'close'
        vis_names = [e.name for e, _ in results["visual"]]
        assert "close" in vis_names
        assert "far" not in vis_names
        # audio (90°) should contain close + mid
        aud_names = [e.name for e, _ in results["audio"]]
        assert "close" in aud_names
        assert "mid" in aud_names

    def test_merge_union(self):
        multi = MultiHeadAttention(self._make_heads())
        close = SceneEntity("close", _y_rot(10))
        mid = SceneEntity("mid", _y_rot(50))
        multi.update(IDENTITY, [close, mid])
        merged = multi.merge_results(strategy="union")
        names = {e.name for e in merged}
        # union: anything attended by any head
        assert "close" in names

    def test_merge_intersection(self):
        multi = MultiHeadAttention(self._make_heads())
        close = SceneEntity("close", _y_rot(5))
        mid = SceneEntity("mid", _y_rot(50))
        multi.update(IDENTITY, [close, mid])
        merged = multi.merge_results(strategy="intersection")
        names = {e.name for e in merged}
        # 'close' is within all cones; 'mid' is NOT within haptic (15°)
        assert "close" in names
        assert "mid" not in names

    def test_merge_unknown_strategy_raises(self):
        multi = MultiHeadAttention(self._make_heads())
        multi.update(IDENTITY, [])
        with pytest.raises(ValueError):
            multi.merge_results(strategy="magic")

    def test_merge_without_update_returns_empty(self):
        multi = MultiHeadAttention(self._make_heads())
        assert multi.merge_results() == []

    def test_update_repoints_all_cones(self):
        multi = MultiHeadAttention(self._make_heads())
        new_dir = _y_rot(45)
        ent = SceneEntity("target", _y_rot(50))
        results = multi.update(new_dir, [ent])
        # 'target' is 5° from new_dir → should be inside visual (30°)
        vis_names = [e.name for e, _ in results["visual"]]
        assert "target" in vis_names


# ---------------------------------------------------------------------------
# Multi-observer with MultiHeadAttention
# ---------------------------------------------------------------------------


class TestMultiObserverMultiHead:
    """Tests for MultiHeadAttention used with different observer origins."""

    def _make_heads(self, origin=IDENTITY):
        return {
            "visual": AttentionCone(origin, half_angle=np.radians(30), falloff="linear"),
            "audio": AttentionCone(origin, half_angle=np.radians(90), falloff="cosine"),
        }

    def test_update_with_different_observer_origins(self):
        multi = MultiHeadAttention(self._make_heads())
        e1 = SceneEntity("front", _y_rot(5))
        e2 = SceneEntity("side", _y_rot(85))

        # Observer 1 at identity — should see 'front' in visual, both in audio
        r1 = multi.update(IDENTITY, [e1, e2])
        r1_vis = [e.name for e, _ in r1["visual"]]
        r1_aud = [e.name for e, _ in r1["audio"]]
        assert "front" in r1_vis
        assert "side" not in r1_vis
        assert "front" in r1_aud

        # Observer 2 at 90° — should see 'side' in visual, not 'front'
        r2 = multi.update(_y_rot(90), [e1, e2])
        r2_vis = [e.name for e, _ in r2["visual"]]
        assert "side" in r2_vis
        assert "front" not in r2_vis
