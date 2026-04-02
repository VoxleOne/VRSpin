"""Tests for the plaza visualization data-generation logic.

Tests the pure-function components of :mod:`examples.plaza_visualization`
without requiring a display — all rendering uses the ``Agg`` backend.
"""

from __future__ import annotations

import os
import tempfile

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

# Ensure the examples package is importable
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from examples.plaza_visualization import (
    ENTITY_POSITIONS,
    VisualizationState,
    compute_plaza_state,
    render_frame,
    generate_demo_frames,
    _strength_bar,
)


# ---------------------------------------------------------------------------
# compute_plaza_state
# ---------------------------------------------------------------------------


class TestComputePlazaState:
    """Tests for :func:`compute_plaza_state`."""

    def test_facing_north_highlights_fountain(self):
        state = compute_plaza_state(0.0)
        assert "Fountain" in state.active_objects

    def test_facing_north_does_not_highlight_market(self):
        state = compute_plaza_state(0.0)
        assert "MarketStand" not in state.active_objects

    def test_facing_east_highlights_market(self):
        state = compute_plaza_state(-70.0)
        assert "MarketStand" in state.active_objects

    def test_facing_nw_shows_vr_art_panel(self):
        state = compute_plaza_state(70.0)
        assert "VR Art" in state.visible_panels

    def test_facing_west_shows_digital_sculpture(self):
        state = compute_plaza_state(85.0)
        assert "Digital Sculpture" in state.visible_panels

    def test_facing_north_no_panels_visible(self):
        state = compute_plaza_state(0.0)
        assert state.visible_panels == []

    def test_audio_playing_when_facing_north(self):
        state = compute_plaza_state(0.0)
        assert "FountainAmbience" in state.audio_playing

    def test_npc_states_populated(self):
        state = compute_plaza_state(0.0)
        assert "Elena" in state.npc_states
        assert "Kai" in state.npc_states

    def test_haptic_strength_when_looking_directly(self):
        state = compute_plaza_state(0.0)
        assert "Fountain" in state.haptic_strengths

    def test_visual_strengths_populated(self):
        state = compute_plaza_state(0.0)
        assert state.visual_strengths  # should have at least one entry

    def test_different_angles_produce_different_states(self):
        state_n = compute_plaza_state(0.0)
        state_e = compute_plaza_state(-70.0)
        assert state_n.active_objects != state_e.active_objects

    def test_cone_toggles_propagated(self):
        state = compute_plaza_state(0.0, show_visual=False, show_audio=False)
        assert not state.show_visual
        assert not state.show_audio
        assert state.show_haptic  # default is True

    def test_full_rotation_coverage(self):
        """Ensure compute_plaza_state works for a full 360° sweep."""
        for angle in range(0, 360, 30):
            state = compute_plaza_state(float(angle))
            assert isinstance(state, VisualizationState)
            assert state.user_yaw_deg == float(angle)

    def test_spinstep_forward_vector_populated(self):
        """forward_vector_from_quaternion produces a valid gaze direction."""
        state = compute_plaza_state(0.0)
        fwd = state.user_forward_vector
        assert len(fwd) == 3
        # Facing north (0° yaw) the forward vector should point along -Z or +Z
        length = (fwd[0] ** 2 + fwd[1] ** 2 + fwd[2] ** 2) ** 0.5
        assert abs(length - 1.0) < 0.01  # unit vector

    def test_spinstep_tree_traversal_populated(self):
        """QuaternionDepthIterator visits nodes in the scene tree."""
        state = compute_plaza_state(0.0)
        assert isinstance(state.tree_attended_names, list)
        # The tree has the root "plaza" plus zone nodes
        assert len(state.tree_attended_names) > 0

    def test_spinstep_entity_distances_populated(self):
        """quaternion_distance produces per-entity angular distances."""
        state = compute_plaza_state(0.0)
        assert "Fountain" in state.entity_distances_deg
        assert "Elena" in state.entity_distances_deg
        # Fountain at north should be near 0° when facing north
        assert state.entity_distances_deg["Fountain"] < 10.0

    def test_spinstep_relative_spins_populated(self):
        """get_relative_spin produces NPC rotation deltas."""
        state = compute_plaza_state(0.0)
        assert "Elena" in state.npc_relative_spins_deg
        assert "Kai" in state.npc_relative_spins_deg
        # Elena (north) should have small relative spin when user faces north
        assert state.npc_relative_spins_deg["Elena"] < 20.0

    def test_spinstep_distances_change_with_orientation(self):
        """quaternion_distance values vary when user rotates."""
        state_n = compute_plaza_state(0.0)
        state_e = compute_plaza_state(-70.0)
        # Fountain distance should increase when turning east
        assert state_e.entity_distances_deg["Fountain"] > state_n.entity_distances_deg["Fountain"]


# ---------------------------------------------------------------------------
# render_frame
# ---------------------------------------------------------------------------


class TestRenderFrame:
    """Tests for :func:`render_frame` (headless / Agg backend)."""

    def test_render_returns_figure(self):
        state = compute_plaza_state(0.0)
        fig = render_frame(state)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_render_saves_to_file(self):
        state = compute_plaza_state(45.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_frame.png")
            render_frame(state, filepath=path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 1000  # non-trivial PNG

    def test_render_with_all_cones_off(self):
        state = compute_plaza_state(
            0.0, show_visual=False, show_audio=False, show_haptic=False,
        )
        fig = render_frame(state)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


# ---------------------------------------------------------------------------
# generate_demo_frames
# ---------------------------------------------------------------------------


class TestGenerateDemoFrames:
    """Tests for :func:`generate_demo_frames`."""

    def test_generates_correct_number_of_frames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_demo_frames(tmpdir, angles=[0.0, 90.0])
            assert len(paths) == 2
            for p in paths:
                assert os.path.exists(p)

    def test_default_angles_generate_eight_frames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_demo_frames(tmpdir)
            assert len(paths) == 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for helper utilities."""

    def test_strength_bar_zero(self):
        bar = _strength_bar(0.0)
        assert "0.00" in bar

    def test_strength_bar_one(self):
        bar = _strength_bar(1.0)
        assert "1.00" in bar
        assert "█" in bar

    def test_strength_bar_half(self):
        bar = _strength_bar(0.5, width=10)
        assert "0.50" in bar

    def test_entity_positions_complete(self):
        """All expected entities have 2D positions defined."""
        expected = {
            "Fountain", "Elena", "FountainAmbience", "VR Art",
            "Digital Sculpture", "MarketStand", "Kai", "MarketMusic",
        }
        assert expected == set(ENTITY_POSITIONS.keys())
