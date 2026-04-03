"""Tests for the VRSpin "Look & Interact" demo package.

Covers the core orientation-cone mechanics, entity state transitions,
NPC attention-machine behaviour, and the VirtualPlaza simulation tick.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from vrspin import (
    AttentionCone,
    AudioSource,
    InteractiveObject,
    KnowledgePanel,
    NPC,
    NPCState,
    PanelPage,
    PlazaEvent,
    VRUser,
    VirtualPlaza,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _y_rot(deg: float) -> np.ndarray:
    """Return a unit quaternion for a yaw of *deg* degrees around +Y."""
    return R.from_euler("y", deg, degrees=True).as_quat()


IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


# ===========================================================================
# AttentionCone
# ===========================================================================


class TestAttentionCone:
    def test_construction_normalises_quaternion(self):
        cone = AttentionCone([0, 0, 0, 2], half_angle_rad=np.deg2rad(30))
        assert np.isclose(np.linalg.norm(cone.orientation), 1.0)

    def test_zero_quaternion_raises(self):
        with pytest.raises(ValueError):
            AttentionCone([0, 0, 0, 0], half_angle_rad=0.5)

    def test_identical_orientation_is_in_cone(self):
        cone = AttentionCone(IDENTITY, half_angle_rad=np.deg2rad(30))
        assert cone.is_in_cone(IDENTITY)

    def test_orientation_just_inside_cone(self):
        cone = AttentionCone(IDENTITY, half_angle_rad=np.deg2rad(30))
        slightly_off = _y_rot(20.0)
        assert cone.is_in_cone(slightly_off)

    def test_orientation_outside_cone(self):
        cone = AttentionCone(IDENTITY, half_angle_rad=np.deg2rad(30))
        far_away = _y_rot(90.0)
        assert not cone.is_in_cone(far_away)

    def test_zero_target_not_in_cone(self):
        cone = AttentionCone(IDENTITY, half_angle_rad=np.deg2rad(60))
        assert not cone.is_in_cone([0, 0, 0, 0])

    def test_update_orientation_changes_forward(self):
        cone = AttentionCone(IDENTITY, half_angle_rad=np.deg2rad(30))
        new_q = _y_rot(90.0)
        cone.update_orientation(new_q)
        assert np.allclose(cone.orientation, new_q / np.linalg.norm(new_q), atol=1e-6)

    def test_get_forward_vector_identity(self):
        cone = AttentionCone(IDENTITY, half_angle_rad=np.deg2rad(45))
        fwd = cone.get_forward_vector()
        assert np.allclose(fwd, [0.0, 0.0, -1.0], atol=1e-6)

    def test_angular_distance_identity(self):
        cone = AttentionCone(IDENTITY, half_angle_rad=np.deg2rad(45))
        assert np.isclose(cone.angular_distance_to(IDENTITY), 0.0, atol=1e-6)

    def test_angular_distance_90_degrees(self):
        cone = AttentionCone(IDENTITY, half_angle_rad=np.deg2rad(45))
        dist = cone.angular_distance_to(_y_rot(90.0))
        assert np.isclose(dist, np.deg2rad(90.0), atol=1e-5)

    def test_filter_within_cone_uses_discrete_orientation_set(self):
        from spinstep import DiscreteOrientationSet

        cone = AttentionCone(IDENTITY, half_angle_rad=np.deg2rad(45))
        dos = DiscreteOrientationSet(np.array([IDENTITY, _y_rot(30), _y_rot(90)]))
        indices = cone.filter_within_cone(dos)
        # identity (0°) and 30° should be inside; 90° should be outside
        assert 0 in indices
        assert 1 in indices
        assert 2 not in indices


# ===========================================================================
# VRUser
# ===========================================================================


class TestVRUser:
    def test_default_orientation_is_identity(self):
        user = VRUser("Test")
        assert np.allclose(user.orientation, IDENTITY)

    def test_invalid_orientation_raises(self):
        with pytest.raises(ValueError):
            VRUser("Bad", orientation=[0, 0, 0, 0])

    def test_set_orientation_updates_all_cones(self):
        user = VRUser("Alice")
        q = _y_rot(45.0)
        user.set_orientation(q)
        assert np.allclose(user.orientation, q / np.linalg.norm(q), atol=1e-6)
        assert np.allclose(user.visual_cone.orientation, user.orientation)
        assert np.allclose(user.audio_cone.orientation, user.orientation)
        assert np.allclose(user.haptic_cone.orientation, user.orientation)

    def test_sees_aligned_object(self):
        user = VRUser("Bob")
        assert user.sees(IDENTITY)  # identity is directly ahead

    def test_sees_returns_false_for_far_object(self):
        user = VRUser("Bob")
        assert not user.sees(_y_rot(90.0))

    def test_hears_wider_than_sees(self):
        user = VRUser("Carol")
        q = _y_rot(80.0)  # 80° — outside visual (60°), inside audio (120°)
        assert not user.sees(q)
        assert user.hears(q)

    def test_feels_narrower_than_sees(self):
        user = VRUser("Dave")
        q = _y_rot(40.0)  # 40° — inside visual (60°), outside haptic (30°)
        assert user.sees(q)
        assert not user.feels(q)

    def test_rotate_by_accumulates(self):
        user = VRUser("Eve")
        delta = _y_rot(30.0)
        user.rotate_by(delta)
        user.rotate_by(delta)
        # Two 30° steps should produce ~60°
        fwd = user.get_forward_vector()
        expected = R.from_euler("y", 60.0, degrees=True).apply([0, 0, -1])
        assert np.allclose(fwd, expected, atol=1e-5)

    def test_cone_for_valid_modalities(self):
        user = VRUser("Frank")
        assert user.cone_for("visual") is user.visual_cone
        assert user.cone_for("audio") is user.audio_cone
        assert user.cone_for("haptic") is user.haptic_cone

    def test_cone_for_invalid_raises(self):
        user = VRUser("Grace")
        with pytest.raises(KeyError):
            user.cone_for("smell")

    def test_repr_contains_name(self):
        user = VRUser("Henry")
        assert "Henry" in repr(user)


# ===========================================================================
# InteractiveObject
# ===========================================================================


class TestInteractiveObject:
    def test_initial_state(self):
        obj = InteractiveObject("Fountain", IDENTITY)
        assert not obj.highlighted
        assert not obj.active

    def test_activate_sets_flags(self):
        obj = InteractiveObject("Fountain", IDENTITY)
        obj.activate()
        assert obj.highlighted
        assert obj.active

    def test_deactivate_clears_flags(self):
        obj = InteractiveObject("Fountain", IDENTITY)
        obj.activate()
        obj.deactivate()
        assert not obj.highlighted
        assert not obj.active

    def test_node_name_matches(self):
        obj = InteractiveObject("Fountain", IDENTITY)
        assert obj.name == obj.node.name == "Fountain"

    def test_orientation_matches_node(self):
        q = _y_rot(15.0)
        obj = InteractiveObject("Post", q)
        assert np.allclose(obj.orientation, obj.node.orientation, atol=1e-6)


# ===========================================================================
# AudioSource
# ===========================================================================


class TestAudioSource:
    def test_initial_state(self):
        a = AudioSource("Music", IDENTITY)
        assert not a.playing
        assert a.volume == 0.0

    def test_start_uses_base_volume(self):
        a = AudioSource("Music", IDENTITY, base_volume=0.6)
        a.start()
        assert a.playing
        assert np.isclose(a.volume, 0.6)

    def test_start_with_explicit_volume(self):
        a = AudioSource("Music", IDENTITY, base_volume=0.8)
        a.start(0.3)
        assert np.isclose(a.volume, 0.3)

    def test_stop(self):
        a = AudioSource("Music", IDENTITY)
        a.start()
        a.stop()
        assert not a.playing
        assert a.volume == 0.0

    def test_set_volume_clamped(self):
        a = AudioSource("Music", IDENTITY)
        a.set_volume(2.0)
        assert a.volume == 1.0
        a.set_volume(-1.0)
        assert a.volume == 0.0


# ===========================================================================
# KnowledgePanel
# ===========================================================================


class TestKnowledgePanel:
    def _make_panel(self) -> KnowledgePanel:
        return KnowledgePanel(
            "Test Panel",
            IDENTITY,
            pages=[
                PanelPage("Page 1", "Content 1"),
                PanelPage("Page 2", "Content 2"),
            ],
        )

    def test_initial_state(self):
        panel = self._make_panel()
        assert not panel.visible
        assert panel.current_page == 0

    def test_show_sets_visible(self):
        panel = self._make_panel()
        panel.show()
        assert panel.visible
        assert panel.current_page == 0

    def test_hide(self):
        panel = self._make_panel()
        panel.show()
        panel.hide()
        assert not panel.visible

    def test_advance_page(self):
        panel = self._make_panel()
        assert panel.advance_page()
        assert panel.current_page == 1

    def test_advance_page_at_last_returns_false(self):
        panel = self._make_panel()
        panel.advance_page()
        assert not panel.advance_page()  # already at last page

    def test_current_content_returns_page(self):
        panel = self._make_panel()
        assert panel.current_content.heading == "Page 1"
        panel.advance_page()
        assert panel.current_content.heading == "Page 2"

    def test_no_pages_current_content_is_none(self):
        panel = KnowledgePanel("Empty", IDENTITY)
        assert panel.current_content is None


# ===========================================================================
# NPC
# ===========================================================================


class TestNPC:
    def test_initial_state(self):
        npc = NPC("Elena", IDENTITY)
        assert npc.state == NPCState.IDLE
        assert npc.noticed_users == []

    def test_tick_user_in_cone_triggers_noticing(self):
        npc = NPC("Elena", IDENTITY, perception_half_angle=np.deg2rad(150))
        user = VRUser("Maya")
        user.set_orientation(IDENTITY)
        events = npc.tick(user)
        assert any("notices" in e for e in events)
        assert npc.state in (NPCState.NOTICING, NPCState.ENGAGED, NPCState.SPEAKING)

    def test_tick_user_outside_cone_stays_idle(self):
        npc = NPC("Elena", IDENTITY, perception_half_angle=np.deg2rad(10))
        user = VRUser("Maya")
        user.set_orientation(_y_rot(90.0))  # facing away
        events = npc.tick(user)
        assert npc.state == NPCState.IDLE
        assert events == []

    def test_user_leaves_cone_resets_to_idle(self):
        npc = NPC("Elena", IDENTITY, perception_half_angle=np.deg2rad(150))
        user = VRUser("Maya")
        # First tick: user in cone
        npc.tick(user)
        assert user.name in npc.noticed_users
        # Second tick: user turns away
        user.set_orientation(_y_rot(170.0))
        npc.tick(user)
        assert npc.state == NPCState.IDLE
        assert user.name not in npc.noticed_users

    def test_npc_slerp_converges(self):
        npc = NPC("Elena", IDENTITY, perception_half_angle=np.deg2rad(150), slerp_speed=0.5)
        user = VRUser("Maya")
        for _ in range(20):
            npc.tick(user)
        # After many ticks the NPC should have nearly converged
        angle = (
            R.from_quat(npc.orientation).inv()
            * R.from_quat(R.from_quat(user.orientation).inv().as_quat())
        ).magnitude()
        assert angle < np.deg2rad(10.0)

    def test_repr_contains_name(self):
        npc = NPC("Kai", IDENTITY)
        assert "Kai" in repr(npc)


# ===========================================================================
# VirtualPlaza
# ===========================================================================


class TestVirtualPlaza:
    def test_construction_builds_all_entities(self):
        plaza = VirtualPlaza()
        assert len(plaza.objects) == 2
        assert len(plaza.npcs) == 2
        assert len(plaza.audio_sources) == 2
        assert len(plaza.knowledge_panels) == 2

    def test_root_is_spinstep_node(self):
        from spinstep import Node

        plaza = VirtualPlaza()
        assert isinstance(plaza.root, Node)
        assert plaza.root.name == "plaza"

    def test_tick_returns_events(self):
        plaza = VirtualPlaza()
        user = VRUser("Maya")
        events = plaza.tick(user)
        assert isinstance(events, list)
        for e in events:
            assert isinstance(e, PlazaEvent)

    def test_tick_facing_north_highlights_fountain(self):
        plaza = VirtualPlaza()
        user = VRUser("Maya")
        user.set_orientation(_y_rot(0.0))
        events = plaza.tick(user)
        visual_sources = [e.source for e in events if e.modality == "visual"]
        assert "Fountain" in visual_sources

    def test_tick_facing_north_starts_fountain_audio(self):
        plaza = VirtualPlaza()
        user = VRUser("Maya")
        user.set_orientation(_y_rot(0.0))
        plaza.tick(user)
        fountain_audio = plaza.get_audio("FountainAmbience")
        assert fountain_audio is not None
        assert fountain_audio.playing

    def test_tick_facing_west_shows_knowledge_panel(self):
        plaza = VirtualPlaza()
        user = VRUser("Maya")
        user.set_orientation(_y_rot(85.0))  # west zone (85°)
        plaza.tick(user)
        panel = plaza.get_panel("Digital Sculpture")
        assert panel is not None
        assert panel.visible

    def test_tick_facing_east_engages_market_stand(self):
        plaza = VirtualPlaza()
        user = VRUser("Maya")
        user.set_orientation(_y_rot(-70.0))  # east zone (-70°)
        events = plaza.tick(user)
        visual_sources = [e.source for e in events if e.modality == "visual"]
        assert "MarketStand" in visual_sources

    def test_tick_count_increments(self):
        plaza = VirtualPlaza()
        user = VRUser("Test")
        assert plaza.tick_count == 0
        plaza.tick(user)
        assert plaza.tick_count == 1
        plaza.tick(user)
        assert plaza.tick_count == 2

    def test_get_object_returns_correct_entity(self):
        plaza = VirtualPlaza()
        fountain = plaza.get_object("Fountain")
        assert fountain is not None
        assert fountain.name == "Fountain"

    def test_get_object_unknown_returns_none(self):
        plaza = VirtualPlaza()
        assert plaza.get_object("DoesNotExist") is None

    def test_get_npc_returns_correct_entity(self):
        plaza = VirtualPlaza()
        elena = plaza.get_npc("Elena")
        assert elena is not None
        assert elena.name == "Elena"

    def test_plaza_event_str_contains_modality(self):
        e = PlazaEvent(1, "visual", "Fountain", "Fountain highlighted")
        assert "VISUAL" in str(e)
        assert "Fountain" in str(e)

    def test_object_deactivates_when_user_turns_away(self):
        plaza = VirtualPlaza()
        user = VRUser("Test")
        # Look north — activates fountain
        user.set_orientation(_y_rot(0.0))
        plaza.tick(user)
        fountain = plaza.get_object("Fountain")
        assert fountain.active
        # Turn far away — should deactivate
        user.set_orientation(_y_rot(90.0))
        plaza.tick(user)
        assert not fountain.active

    def test_audio_attenuates_outside_cone(self):
        plaza = VirtualPlaza()
        user = VRUser("Test")
        # Start audio by facing north
        user.set_orientation(_y_rot(0.0))
        plaza.tick(user)
        fountain_audio = plaza.get_audio("FountainAmbience")
        assert fountain_audio.playing
        # Turn 150° — outside audio cone (120° half-angle)
        user.set_orientation(_y_rot(150.0))
        plaza.tick(user)
        assert fountain_audio.volume < fountain_audio.base_volume