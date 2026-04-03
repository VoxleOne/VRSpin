#!/usr/bin/env python3
"""Interactive VRSpin Plaza Visualization — Desktop Demo (no VR headset needed).

A matplotlib-based top-down view of the VRSpin virtual plaza that visualizes
orientation-driven attention cones, entity highlighting, NPC awareness, and
spatial audio gain in real time.

All quaternion operations use **SpinStep** primitives directly:

* :func:`~spinstep.utils.quaternion_from_euler` — convert user yaw to quaternion
* :func:`~spinstep.utils.forward_vector_from_quaternion` — extract gaze direction
* :func:`~spinstep.utils.quaternion_distance` — angular distance between orientations
* :func:`~spinstep.utils.is_within_angle_threshold` — cone membership test
* :func:`~spinstep.utils.batch_quaternion_angle` — vectorised distance matrix
* :func:`~spinstep.utils.get_relative_spin` — NPC-to-user rotation delta
* :class:`~spinstep.Node` — scene-tree nodes
* :class:`~spinstep.QuaternionDepthIterator` — orientation-aligned tree traversal

Controls::

    Left/Right arrow keys   Rotate user orientation (yaw)
    Up/Down arrow keys      Adjust visual cone half-angle
    1 / 2 / 3               Toggle visual / audio / haptic cone visibility
    R                       Reset user to face north (0°)
    Q / Escape              Quit

Run::

    python examples/plaza_visualization.py

Requirements::

    pip install matplotlib numpy scipy
    pip install -e .  # install vrspin (pulls in spinstep)

"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt


def _ensure_agg_backend() -> None:
    """Switch to the non-interactive Agg backend if no display is available."""
    current = matplotlib.get_backend()
    if current.lower() not in ("agg",):
        try:
            matplotlib.use("Agg")
        except Exception:
            pass


import matplotlib.patches as mpatches
from matplotlib.patches import Wedge
import numpy as np

# ---------------------------------------------------------------------------
# SpinStep imports — the quaternion-driven orientation framework that powers
# all perception mechanics in this visualization.
# ---------------------------------------------------------------------------
from spinstep import Node, QuaternionDepthIterator
from spinstep.utils import (
    quaternion_from_euler,
    forward_vector_from_quaternion,
    quaternion_distance,
    is_within_angle_threshold,
    batch_quaternion_angle,
    get_relative_spin,
)

# VRSpin classes — built on top of SpinStep primitives.
from vrspin import (
    AttentionCone,
    VirtualPlaza,
    VRUser,
    PlazaEvent,
)


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

COLORS = {
    "background": "#1a1a2e",
    "grid": "#2a2a4a",
    "user": "#00ff88",
    "user_arrow": "#00ff88",
    "visual_cone": "#4ade80",
    "audio_cone": "#fbbf24",
    "haptic_cone": "#c084fc",
    "object_idle": "#64748b",
    "object_active": "#22d3ee",
    "npc_idle": "#f87171",
    "npc_noticing": "#fb923c",
    "npc_engaged": "#4ade80",
    "npc_speaking": "#60a5fa",
    "audio_source": "#fbbf24",
    "panel_hidden": "#94a3b8",
    "panel_visible": "#818cf8",
    "text": "#e2e8f0",
    "event_bg": "#0f172a",
}

# Maximum number of recent events to display in the info panel.
_MAX_DISPLAYED_EVENTS = 5


# ---------------------------------------------------------------------------
# Plaza layout — 2D positions for entities
# ---------------------------------------------------------------------------

# Positions chosen to match the plaza scene description:
#   North (0°)  — Fountain + Elena + Fountain Ambience
#   NW (70°)    — VR Art panel
#   West (85°)  — Digital Sculpture panel
#   East (-70°) — Market Stand + Kai + Market Music

ENTITY_POSITIONS: Dict[str, Tuple[float, float]] = {
    "Fountain":          (0.0,  4.0),
    "Elena":             (1.5,  3.5),
    "FountainAmbience":  (-1.0, 4.5),
    "VR Art":            (-3.8, 3.0),
    "Digital Sculpture": (-4.5, 1.0),
    "MarketStand":       (3.8,  3.0),
    "Kai":               (4.5,  2.0),
    "MarketMusic":       (3.5,  4.0),
}

ENTITY_ICONS = {
    "Fountain": "◆",
    "Elena": "♀",
    "FountainAmbience": "♫",
    "VR Art": "▣",
    "Digital Sculpture": "▣",
    "MarketStand": "◆",
    "Kai": "♂",
    "MarketMusic": "♫",
}


# ---------------------------------------------------------------------------
# Data model for visualization state
# ---------------------------------------------------------------------------


@dataclass
class VisualizationState:
    """Snapshot of the plaza state for one frame of visualization."""

    user_yaw_deg: float = 0.0
    user_position: Tuple[float, float] = (0.0, 0.0)

    # Cone visibility toggles
    show_visual: bool = True
    show_audio: bool = True
    show_haptic: bool = True

    # Entity states from last tick
    active_objects: List[str] = field(default_factory=list)
    visible_panels: List[str] = field(default_factory=list)
    npc_states: Dict[str, str] = field(default_factory=dict)
    audio_playing: Dict[str, float] = field(default_factory=dict)
    events: List[str] = field(default_factory=list)

    # Attention strengths per entity
    visual_strengths: Dict[str, float] = field(default_factory=dict)
    audio_strengths: Dict[str, float] = field(default_factory=dict)
    haptic_strengths: Dict[str, float] = field(default_factory=dict)

    # SpinStep primitive results — showcasing the underlying framework
    tree_attended_names: List[str] = field(default_factory=list)
    entity_distances_deg: Dict[str, float] = field(default_factory=dict)
    npc_relative_spins_deg: Dict[str, float] = field(default_factory=dict)
    user_forward_vector: Tuple[float, float, float] = (0.0, 0.0, -1.0)


# ---------------------------------------------------------------------------
# Core visualization logic
# ---------------------------------------------------------------------------


def compute_plaza_state(
    user_yaw_deg: float,
    show_visual: bool = True,
    show_audio: bool = True,
    show_haptic: bool = True,
) -> VisualizationState:
    """Run one plaza tick and capture the full state for visualization.

    Uses SpinStep primitives throughout:

    * :func:`~spinstep.utils.quaternion_from_euler` for user orientation
    * :func:`~spinstep.utils.forward_vector_from_quaternion` for gaze direction
    * :func:`~spinstep.utils.quaternion_distance` for per-entity distances
    * :func:`~spinstep.utils.is_within_angle_threshold` (via AttentionCone)
    * :func:`~spinstep.utils.batch_quaternion_angle` (via AttentionCone)
    * :func:`~spinstep.utils.get_relative_spin` for NPC rotation deltas
    * :class:`~spinstep.QuaternionDepthIterator` for scene-tree traversal

    Args:
        user_yaw_deg: User's yaw angle in degrees (0 = north/+Z).
        show_visual: Whether visual cone is displayed.
        show_audio: Whether audio cone is displayed.
        show_haptic: Whether haptic cone is displayed.

    Returns:
        A :class:`VisualizationState` with all entity states populated.
    """
    plaza = VirtualPlaza()
    user = VRUser("Viewer")

    # SpinStep: quaternion_from_euler — convert headset yaw to quaternion.
    user_quat = quaternion_from_euler(
        [user_yaw_deg, 0, 0], order="yxz", degrees=True
    )
    user.set_orientation(user_quat)

    # SpinStep: forward_vector_from_quaternion — extract gaze direction.
    fwd = forward_vector_from_quaternion(user_quat)

    # Run two ticks so NPCs can transition
    plaza.tick(user)
    events_list = plaza.tick(user)

    state = VisualizationState(
        user_yaw_deg=user_yaw_deg,
        show_visual=show_visual,
        show_audio=show_audio,
        show_haptic=show_haptic,
        user_forward_vector=(float(fwd[0]), float(fwd[1]), float(fwd[2])),
    )

    # SpinStep: QuaternionDepthIterator — traverse the Node tree using
    # the user's orientation as the rotation step.  May yield no results
    # when the tree is empty or the angle threshold excludes all children.
    try:
        for node in QuaternionDepthIterator(
            plaza.root, user_quat, angle_threshold=np.deg2rad(45.0),
        ):
            state.tree_attended_names.append(node.name)
    except StopIteration:
        pass

    # SpinStep: quaternion_distance — per-entity angular distances.
    all_entities = (
        [(obj.name, obj.orientation) for obj in plaza.objects]
        + [(npc.name, npc.orientation) for npc in plaza.npcs]
        + [(a.name, a.orientation) for a in plaza.audio_sources]
        + [(p.name, p.orientation) for p in plaza.knowledge_panels]
    )
    for name, quat in all_entities:
        dist_rad = float(quaternion_distance(user_quat, quat))
        state.entity_distances_deg[name] = float(np.rad2deg(dist_rad))

    # SpinStep: get_relative_spin — compute rotation delta NPC→user.
    user_node = Node("user", user_quat)
    for npc in plaza.npcs:
        rel_quat = get_relative_spin(npc.node, user_node)
        rel_angle_deg = float(
            np.rad2deg(quaternion_distance(rel_quat, np.array([0, 0, 0, 1])))
        )
        state.npc_relative_spins_deg[npc.name] = rel_angle_deg

    # Active objects
    for obj in plaza.objects:
        if obj.active:
            state.active_objects.append(obj.name)
        # Compute per-entity visual strength
        strength = user.visual_cone.attenuation(obj.orientation)
        if strength > 0:
            state.visual_strengths[obj.name] = strength

    # Knowledge panels
    for panel in plaza.knowledge_panels:
        if panel.visible:
            state.visible_panels.append(panel.name)
        strength = user.visual_cone.attenuation(panel.orientation)
        if strength > 0:
            state.visual_strengths[panel.name] = strength

    # NPC states
    for npc in plaza.npcs:
        state.npc_states[npc.name] = npc.state.name.lower()
        strength = user.visual_cone.attenuation(npc.orientation)
        if strength > 0:
            state.visual_strengths[npc.name] = strength

    # Audio
    for audio in plaza.audio_sources:
        if audio.playing:
            state.audio_playing[audio.name] = audio.volume
        gain = user.audio_cone.attenuation(audio.orientation)
        if gain > 0:
            state.audio_strengths[audio.name] = gain

    # Haptic strengths
    for obj in plaza.objects:
        strength = user.haptic_cone.attenuation(obj.orientation)
        if strength > 0:
            state.haptic_strengths[obj.name] = strength

    # Events
    state.events = [str(e) for e in events_list[:_MAX_DISPLAYED_EVENTS]]

    return state


def render_frame(state: VisualizationState, filepath: Optional[str] = None) -> plt.Figure:
    """Render one frame of the plaza visualization.

    Args:
        state: Current visualization state.
        filepath: If provided, save the figure to this path instead of
            displaying it.

    Returns:
        The matplotlib Figure.
    """
    fig = plt.figure(figsize=(14, 10), facecolor=COLORS["background"])

    # Main plaza view (left) + info panel (right)
    ax_plaza = fig.add_axes([0.02, 0.05, 0.62, 0.88])
    ax_info = fig.add_axes([0.67, 0.05, 0.31, 0.88])

    _draw_plaza(ax_plaza, state)
    _draw_info_panel(ax_info, state)

    # Title
    fig.text(
        0.5, 0.97,
        "VRSpin Plaza — SpinStep Orientation-Driven Attention Visualization",
        ha="center", va="top",
        fontsize=14, fontweight="bold",
        color=COLORS["text"],
    )
    fig.text(
        0.5, 0.935,
        f"User facing: {state.user_yaw_deg:.0f}°  |  "
        "Controls: ←/→ rotate, 1/2/3 toggle cones, R reset, Q quit",
        ha="center", va="top",
        fontsize=9,
        color="#94a3b8",
    )

    if filepath:
        fig.savefig(filepath, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)

    return fig


def _draw_plaza(ax: plt.Axes, state: VisualizationState) -> None:
    """Draw the top-down plaza view with cones and entities."""
    ax.set_facecolor(COLORS["background"])
    ax.set_xlim(-7, 7)
    ax.set_ylim(-2, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    # Grid
    for x in range(-6, 7, 2):
        ax.axvline(x, color=COLORS["grid"], linewidth=0.3, alpha=0.5)
    for y in range(-1, 7, 2):
        ax.axhline(y, color=COLORS["grid"], linewidth=0.3, alpha=0.5)

    # Plaza boundary circle
    plaza_circle = plt.Circle(
        (0, 2.5), 5.5, fill=False,
        edgecolor="#334155", linewidth=1.5, linestyle="--",
    )
    ax.add_patch(plaza_circle)

    # Cardinal labels
    ax.text(0, 6.7, "N (0°)", ha="center", fontsize=8, color="#64748b")
    ax.text(-6.3, 2.5, "W (85°)", ha="center", fontsize=8, color="#64748b", rotation=90)
    ax.text(6.3, 2.5, "E (−70°)", ha="center", fontsize=8, color="#64748b", rotation=-90)

    # --- Draw attention cones ---
    ux, uy = state.user_position

    # Yaw 0° = north (+Y in our 2D view). Matplotlib wedge angles are
    # measured CCW from +X, so north = 90°. User yaw positive = left in
    # the scene, which is CCW in top-down view.
    base_angle = 90.0 + state.user_yaw_deg  # north + user yaw

    cone_radius = 6.0

    if state.show_audio:
        _draw_cone_wedge(
            ax, ux, uy, cone_radius,
            base_angle, VRUser.AUDIO_HALF_ANGLE,
            COLORS["audio_cone"], alpha=0.08, label="Audio (120°)",
        )
    if state.show_visual:
        _draw_cone_wedge(
            ax, ux, uy, cone_radius,
            base_angle, VRUser.VISUAL_HALF_ANGLE,
            COLORS["visual_cone"], alpha=0.12, label="Visual (60°)",
        )
    if state.show_haptic:
        _draw_cone_wedge(
            ax, ux, uy, cone_radius * 0.6,
            base_angle, VRUser.HAPTIC_HALF_ANGLE,
            COLORS["haptic_cone"], alpha=0.15, label="Haptic (30°)",
        )

    # --- Draw entities ---
    for name, (ex, ey) in ENTITY_POSITIONS.items():
        _draw_entity(ax, name, ex, ey, state)

    # --- Draw user ---
    ax.plot(ux, uy, "o", color=COLORS["user"], markersize=12, zorder=10)
    ax.plot(ux, uy, "o", color=COLORS["background"], markersize=6, zorder=11)

    # Forward direction arrow
    fwd_angle_rad = np.deg2rad(base_angle)
    dx = np.cos(fwd_angle_rad) * 1.2
    dy = np.sin(fwd_angle_rad) * 1.2
    ax.annotate(
        "", xy=(ux + dx, uy + dy), xytext=(ux, uy),
        arrowprops=dict(
            arrowstyle="-|>", color=COLORS["user_arrow"],
            lw=2.5, mutation_scale=15,
        ),
        zorder=12,
    )
    ax.text(ux, uy - 0.6, "USER", ha="center", fontsize=8,
            fontweight="bold", color=COLORS["user"])

    # Legend
    legend_items = []
    if state.show_visual:
        legend_items.append(mpatches.Patch(
            facecolor=COLORS["visual_cone"], alpha=0.3, label="Visual cone (60°)"
        ))
    if state.show_audio:
        legend_items.append(mpatches.Patch(
            facecolor=COLORS["audio_cone"], alpha=0.3, label="Audio cone (120°)"
        ))
    if state.show_haptic:
        legend_items.append(mpatches.Patch(
            facecolor=COLORS["haptic_cone"], alpha=0.3, label="Haptic cone (30°)"
        ))
    if legend_items:
        leg = ax.legend(
            handles=legend_items, loc="lower left",
            fontsize=7, framealpha=0.7,
            facecolor=COLORS["event_bg"], edgecolor="#334155",
            labelcolor=COLORS["text"],
        )


def _draw_cone_wedge(
    ax: plt.Axes, cx: float, cy: float, radius: float,
    center_angle_deg: float, half_angle_rad: float,
    color: str, alpha: float = 0.15, label: str = "",
) -> None:
    """Draw a wedge representing an attention cone on the plaza view."""
    half_angle_deg = np.rad2deg(half_angle_rad)
    theta1 = center_angle_deg - half_angle_deg
    theta2 = center_angle_deg + half_angle_deg

    wedge = Wedge(
        (cx, cy), radius, theta1, theta2,
        facecolor=color, edgecolor=color,
        alpha=alpha, linewidth=1.0,
    )
    ax.add_patch(wedge)

    # Cone edge lines
    for angle in (theta1, theta2):
        rad = np.deg2rad(angle)
        ax.plot(
            [cx, cx + np.cos(rad) * radius],
            [cy, cy + np.sin(rad) * radius],
            color=color, alpha=0.3, linewidth=0.8, linestyle="--",
        )


def _draw_entity(
    ax: plt.Axes, name: str, x: float, y: float,
    state: VisualizationState,
) -> None:
    """Draw a single entity with state-dependent styling."""
    icon = ENTITY_ICONS.get(name, "●")
    is_active = name in state.active_objects
    is_visible_panel = name in state.visible_panels
    npc_state = state.npc_states.get(name)
    audio_vol = state.audio_playing.get(name)

    # Determine colour
    if npc_state:
        npc_color_map = {
            "idle": COLORS["npc_idle"],
            "noticing": COLORS["npc_noticing"],
            "engaged": COLORS["npc_engaged"],
            "speaking": COLORS["npc_speaking"],
        }
        color = npc_color_map.get(npc_state, COLORS["npc_idle"])
        marker_size = 14
    elif is_active:
        color = COLORS["object_active"]
        marker_size = 14
    elif is_visible_panel:
        color = COLORS["panel_visible"]
        marker_size = 13
    elif audio_vol is not None:
        color = COLORS["audio_source"]
        marker_size = 12
    elif name in ("VR Art", "Digital Sculpture"):
        color = COLORS["panel_hidden"]
        marker_size = 11
    else:
        color = COLORS["object_idle"]
        marker_size = 11

    # Glow effect for active entities
    if is_active or is_visible_panel or (npc_state and npc_state != "idle"):
        glow = plt.Circle((x, y), 0.5, facecolor=color, alpha=0.15, edgecolor="none")
        ax.add_patch(glow)

    # Entity marker
    ax.text(x, y, icon, ha="center", va="center",
            fontsize=marker_size, color=color, zorder=8)

    # Label
    label = name
    extra = ""
    if npc_state and npc_state != "idle":
        extra = f" [{npc_state}]"
    elif audio_vol is not None:
        extra = f" (vol:{audio_vol:.2f})"
    elif is_active:
        extra = " ✦"
    elif is_visible_panel:
        extra = " [visible]"

    ax.text(x, y - 0.45, f"{label}{extra}", ha="center", fontsize=6.5,
            color=color, alpha=0.9, zorder=8)

    # Visual strength indicator
    vis = state.visual_strengths.get(name, 0)
    if vis > 0:
        ring = plt.Circle(
            (x, y), 0.35, fill=False,
            edgecolor=COLORS["visual_cone"],
            linewidth=1.0 + vis * 2.0,
            alpha=0.3 + vis * 0.5,
        )
        ax.add_patch(ring)


def _draw_info_panel(ax: plt.Axes, state: VisualizationState) -> None:
    """Draw the information side panel."""
    ax.set_facecolor(COLORS["event_bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y = 0.96
    line_h = 0.032

    def _text(text: str, color: str = COLORS["text"], bold: bool = False,
              size: int = 8, indent: int = 0) -> None:
        nonlocal y
        weight = "bold" if bold else "normal"
        ax.text(0.03 + indent * 0.04, y, text, fontsize=size, color=color,
                fontweight=weight, va="top", transform=ax.transAxes)
        y -= line_h

    # Header
    _text("SCENE STATUS", bold=True, size=10, color="#60a5fa")
    y -= 0.01

    # User info
    _text(f"User Yaw: {state.user_yaw_deg:.0f}°", color=COLORS["user"], bold=True)
    y -= 0.01

    # Cone toggles
    _text("ATTENTION CONES", bold=True, size=9, color="#94a3b8")
    vis_status = "ON" if state.show_visual else "OFF"
    aud_status = "ON" if state.show_audio else "OFF"
    hap_status = "ON" if state.show_haptic else "OFF"
    _text(f"[1] Visual  (60°):  {vis_status}", color=COLORS["visual_cone"], indent=1)
    _text(f"[2] Audio  (120°):  {aud_status}", color=COLORS["audio_cone"], indent=1)
    _text(f"[3] Haptic  (30°):  {hap_status}", color=COLORS["haptic_cone"], indent=1)
    y -= 0.01

    # Objects
    _text("INTERACTIVE OBJECTS", bold=True, size=9, color="#94a3b8")
    for obj_name in ("Fountain", "MarketStand"):
        active = obj_name in state.active_objects
        status = "✦ HIGHLIGHTED" if active else "idle"
        col = COLORS["object_active"] if active else COLORS["object_idle"]
        strength = state.visual_strengths.get(obj_name, 0)
        bar = _strength_bar(strength)
        _text(f"  {obj_name}: {status}  {bar}", color=col, indent=1)
    y -= 0.01

    # NPCs
    _text("NPCs", bold=True, size=9, color="#94a3b8")
    for npc_name in ("Elena", "Kai"):
        npc_st = state.npc_states.get(npc_name, "idle")
        npc_color_map = {
            "idle": COLORS["npc_idle"],
            "noticing": COLORS["npc_noticing"],
            "engaged": COLORS["npc_engaged"],
            "speaking": COLORS["npc_speaking"],
        }
        col = npc_color_map.get(npc_st, COLORS["npc_idle"])
        _text(f"  {npc_name}: {npc_st.upper()}", color=col, indent=1)
    y -= 0.01

    # Audio
    _text("AUDIO SOURCES", bold=True, size=9, color="#94a3b8")
    for audio_name in ("FountainAmbience", "MarketMusic"):
        vol = state.audio_playing.get(audio_name)
        gain = state.audio_strengths.get(audio_name, 0)
        if vol is not None:
            bar = _strength_bar(gain)
            _text(f"  {audio_name}: ♫ vol={vol:.2f}  {bar}", color=COLORS["audio_source"], indent=1)
        else:
            _text(f"  {audio_name}: silent", color="#64748b", indent=1)
    y -= 0.01

    # Knowledge Panels
    _text("KNOWLEDGE PANELS", bold=True, size=9, color="#94a3b8")
    for panel_name in ("VR Art", "Digital Sculpture"):
        visible = panel_name in state.visible_panels
        status = "[VISIBLE]" if visible else "hidden"
        col = COLORS["panel_visible"] if visible else COLORS["panel_hidden"]
        _text(f"  {panel_name}: {status}", color=col, indent=1)
    y -= 0.01

    # Haptic
    _text("HAPTIC FEEDBACK", bold=True, size=9, color="#94a3b8")
    if state.haptic_strengths:
        for name, strength in state.haptic_strengths.items():
            bar = _strength_bar(strength)
            _text(f"  {name}: pulse {bar}", color=COLORS["haptic_cone"], indent=1)
    else:
        _text("  (no objects in haptic range)", color="#64748b", indent=1)
    y -= 0.01

    # Recent events
    if state.events:
        _text("RECENT EVENTS", bold=True, size=9, color="#94a3b8")
        for evt in state.events[:_MAX_DISPLAYED_EVENTS]:
            _text(f"  {evt[:55]}", color="#cbd5e1", indent=1, size=7)
    y -= 0.01

    # SpinStep primitives showcase
    _text("SPINSTEP PRIMITIVES", bold=True, size=9, color="#f472b6")
    fwd = state.user_forward_vector
    _text(
        f"  forward_vector: [{fwd[0]:+.2f}, {fwd[1]:+.2f}, {fwd[2]:+.2f}]",
        color="#f9a8d4", indent=1, size=7,
    )
    tree_count = len(state.tree_attended_names)
    _text(
        f"  QuaternionDepthIterator: {tree_count} nodes visited",
        color="#f9a8d4", indent=1, size=7,
    )
    for name in ("Fountain", "MarketStand", "Elena", "Kai"):
        dist = state.entity_distances_deg.get(name)
        if dist is not None:
            _text(
                f"  quat_distance({name}): {dist:.1f}°",
                color="#f9a8d4", indent=1, size=7,
            )
    for npc_name in ("Elena", "Kai"):
        spin = state.npc_relative_spins_deg.get(npc_name)
        if spin is not None:
            _text(
                f"  relative_spin({npc_name}): {spin:.1f}°",
                color="#f9a8d4", indent=1, size=7,
            )


def _strength_bar(value: float, width: int = 10) -> str:
    """Render a text-based strength bar: ████░░░░░░ 0.72"""
    filled = int(value * width)
    empty = width - filled
    return f"{'█' * filled}{'░' * empty} {value:.2f}"


# ---------------------------------------------------------------------------
# Static image generation (no-headset demo)
# ---------------------------------------------------------------------------


def generate_demo_frames(output_dir: str = ".", angles: Optional[List[float]] = None) -> List[str]:
    """Generate a set of demo visualization frames as PNG files.

    This function creates the demo without requiring any interactive
    display — perfect for CI, documentation, or headless environments.

    Args:
        output_dir: Directory to save PNG files.
        angles: List of yaw angles to render. Defaults to a scenic tour.

    Returns:
        List of saved file paths.
    """
    if angles is None:
        angles = [0.0, 15.0, 45.0, 70.0, 85.0, -30.0, -70.0, 0.0]

    _ensure_agg_backend()

    saved: List[str] = []
    for i, yaw in enumerate(angles):
        state = compute_plaza_state(yaw)
        path = f"{output_dir}/vrspin_plaza_frame_{i:02d}_{yaw:.0f}deg.png"
        render_frame(state, filepath=path)
        saved.append(path)
        print(f"  Saved frame {i}: yaw={yaw:.0f}° → {path}")

    return saved


# ---------------------------------------------------------------------------
# Interactive mode (requires display)
# ---------------------------------------------------------------------------


def run_interactive() -> None:
    """Run the interactive matplotlib visualization.

    Uses keyboard input to control user orientation in real time.
    Falls back to static frame generation if no display is available.
    """
    try:
        matplotlib.use("TkAgg")
    except ImportError:
        pass

    if not _has_display():
        print("No display detected — generating static demo frames instead.")
        print()
        generate_demo_frames(".")
        return

    yaw = [0.0]
    show = {"visual": True, "audio": True, "haptic": True}

    fig = [None]

    def _redraw() -> None:
        if fig[0] is not None:
            plt.close(fig[0])
        state = compute_plaza_state(
            yaw[0],
            show_visual=show["visual"],
            show_audio=show["audio"],
            show_haptic=show["haptic"],
        )
        fig[0] = render_frame(state)
        fig[0].canvas.mpl_connect("key_press_event", _on_key)
        plt.show(block=False)
        fig[0].canvas.draw()

    def _on_key(event) -> None:
        if event.key in ("q", "escape"):
            plt.close("all")
            return
        if event.key == "left":
            yaw[0] = (yaw[0] + 10) % 360
        elif event.key == "right":
            yaw[0] = (yaw[0] - 10) % 360
        elif event.key == "r":
            yaw[0] = 0.0
        elif event.key == "1":
            show["visual"] = not show["visual"]
        elif event.key == "2":
            show["audio"] = not show["audio"]
        elif event.key == "3":
            show["haptic"] = not show["haptic"]
        else:
            return
        _redraw()

    _redraw()
    plt.show()


def _has_display() -> bool:
    """Check whether a graphical display is available."""
    import os
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return True
    # macOS
    if sys.platform == "darwin":
        return True
    return False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point — runs interactive mode or generates static frames."""
    import argparse

    parser = argparse.ArgumentParser(
        description="VRSpin Plaza Visualization — desktop attention-cone demo",
    )
    parser.add_argument(
        "--static", action="store_true",
        help="Generate static PNG frames instead of interactive mode",
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="Output directory for static frames (default: current dir)",
    )
    parser.add_argument(
        "--angles", type=str, default=None,
        help="Comma-separated yaw angles for static mode (e.g. '0,45,90,-70')",
    )
    args = parser.parse_args()

    if args.static:
        angles = None
        if args.angles:
            angles = [float(a.strip()) for a in args.angles.split(",")]
        print("Generating VRSpin plaza visualization frames...")
        print()
        paths = generate_demo_frames(args.output_dir, angles)
        print(f"\nDone — {len(paths)} frames saved.")
    else:
        run_interactive()


if __name__ == "__main__":
    main()
