"""Virtual plaza simulation engine.

The :class:`VirtualPlaza` assembles all scene entities into a SpinStep
:class:`~spinstep.Node` tree and drives the attention-cone mechanics each
simulation tick.

Scene tree layout::

    plaza (root)
    ├── north_zone   [0°]   — fountain / Elena NPC / ambient water audio
    ├── nw_zone     [70°]   — VR Art knowledge panel
    ├── west_zone   [85°]   — Digital Sculpture knowledge panel
    └── east_zone  [-70°]   — market stand / Kai NPC / market music

Orientations are chosen so that the panels and market stand are **outside**
the user's forward visual cone (60° half-angle) when facing north, and only
become visible when the user deliberately turns in their direction.

The :class:`~spinstep.QuaternionDepthIterator` traverses this tree using the
user's orientation as the rotation step, naturally visiting entities that
align with the current gaze direction.
"""

from __future__ import annotations

__all__ = ["VirtualPlaza", "PlazaEvent"]

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from spinstep import Node, QuaternionDepthIterator
from spinstep.utils import quaternion_from_euler, quaternion_distance as _spinstep_quat_distance

from .cone import AttentionCone
from .entities import AudioSource, InteractiveObject, KnowledgePanel, PanelPage
from .npc import NPC
from .user import VRUser


# ---------------------------------------------------------------------------
# Event record
# ---------------------------------------------------------------------------


@dataclass
class PlazaEvent:
    """A timestamped event produced by the plaza simulation.

    Attributes:
        tick: Simulation tick number.
        modality: One of ``"visual"``, ``"audio"``, ``"haptic"``, ``"npc"``,
            ``"knowledge"``, or ``"system"``.
        source: Name of the entity that generated the event.
        message: Human-readable description.
    """

    tick: int
    modality: str
    source: str
    message: str

    def __str__(self) -> str:
        return f"[{self.modality.upper():8s}] ({self.source}) {self.message}"


# ---------------------------------------------------------------------------
# Virtual Plaza
# ---------------------------------------------------------------------------


class VirtualPlaza:
    """An orientation-driven virtual plaza powered by SpinStep.

    The plaza simulates a Horizon-Worlds-style environment where all
    perception and interaction is driven by the user's head orientation.

    Args:
        audio_base_volume: Default volume for audio sources when in cone.

    Attributes:
        root: SpinStep :class:`~spinstep.Node` scene-tree root.
        npcs: All :class:`~vrspin.npc.NPC` instances.
        objects: All :class:`~vrspin.entities.InteractiveObject` instances.
        audio_sources: All :class:`~vrspin.entities.AudioSource` instances.
        knowledge_panels: All :class:`~vrspin.entities.KnowledgePanel` instances.

    Example::

        from scipy.spatial.transform import Rotation as R
        from vrspin import VirtualPlaza, VRUser

        plaza = VirtualPlaza()
        user = VRUser("Alice")
        events = plaza.tick(user)
        for e in events:
            print(e)
    """

    def __init__(self, audio_base_volume: float = 0.8) -> None:
        self._audio_base_volume = audio_base_volume
        self.tick_count: int = 0
        self._build_scene()

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _build_scene(self) -> None:
        """Construct the SpinStep node tree and populate all entities."""

        # ------ orientation helpers ------
        def _y_rot(deg: float) -> np.ndarray:
            """Unit quaternion for a rotation of *deg* degrees around +Y."""
            return quaternion_from_euler([0.0, deg, 0.0], order="xyz")

        north = _y_rot(0.0)        # [0, 0, 0, 1] — fountain / Elena
        nw = _y_rot(70.0)          # 70° left — VR Art panel (outside front FOV)
        west = _y_rot(85.0)        # 85° left — Digital Sculpture panel
        east = _y_rot(-70.0)       # 70° right — market area

        # ------ scene entities ------
        self.objects: List[InteractiveObject] = [
            InteractiveObject("Fountain", north, "A cascading stone fountain at the plaza centre."),
            InteractiveObject("MarketStand", east, "A colourful merchant stall selling VR goods."),
        ]

        self.npcs: List[NPC] = [
            NPC(
                "Elena",
                north,
                greeting="Welcome to the plaza! The fountain has been here for ages.",
                perception_half_angle=np.deg2rad(120.0),
                slerp_speed=0.5,
            ),
            NPC(
                "Kai",
                east,
                greeting="Step right up! Best wares in the metaverse!",
                perception_half_angle=np.deg2rad(100.0),
                slerp_speed=0.5,
            ),
        ]

        self.audio_sources: List[AudioSource] = [
            AudioSource(
                "FountainAmbience",
                north,
                content="soft water sounds",
                base_volume=0.7,
            ),
            AudioSource(
                "MarketMusic",
                east,
                content="lively merchant tune",
                base_volume=0.9,
            ),
        ]

        self.knowledge_panels: List[KnowledgePanel] = [
            KnowledgePanel(
                "VR Art",
                nw,
                pages=[
                    PanelPage(
                        "Virtual Reality Art",
                        "Creating immersive art in virtual worlds — "
                        "orientation shapes what you perceive.",
                        trigger_angle_deg=0.0,
                    ),
                    PanelPage(
                        "Interaction Design in VR",
                        "How head orientation drives UX — menus replaced by looking.",
                        trigger_angle_deg=10.0,
                    ),
                ],
            ),
            KnowledgePanel(
                "Digital Sculpture",
                west,
                pages=[
                    PanelPage(
                        "Digital Sculpture",
                        "3-D sculptural works that exist purely in virtual space.",
                        trigger_angle_deg=0.0,
                    ),
                ],
            ),
        ]

        # ------ SpinStep Node tree ------
        north_zone = Node(
            "north_zone",
            north,
            children=[e.node for e in self.objects if _angle_to(e.orientation, north) < 0.1]
            + [npc.node for npc in self.npcs if _angle_to(npc.orientation, north) < 0.1]
            + [a.node for a in self.audio_sources if _angle_to(a.orientation, north) < 0.1],
        )
        nw_zone = Node(
            "northwest_zone",
            nw,
            children=[p.node for p in self.knowledge_panels if _angle_to(p.orientation, nw) < 0.1],
        )
        west_zone = Node(
            "west_zone",
            west,
            children=[p.node for p in self.knowledge_panels if _angle_to(p.orientation, west) < 0.1],
        )
        east_zone = Node(
            "east_zone",
            east,
            children=[e.node for e in self.objects if _angle_to(e.orientation, east) < 0.1]
            + [npc.node for npc in self.npcs if _angle_to(npc.orientation, east) < 0.1]
            + [a.node for a in self.audio_sources if _angle_to(a.orientation, east) < 0.1],
        )

        self.root = Node(
            "plaza",
            [0, 0, 0, 1],
            children=[north_zone, nw_zone, west_zone, east_zone],
        )

        # Fast name→entity lookups
        self._objects_by_name: Dict[str, InteractiveObject] = {o.name: o for o in self.objects}
        self._npcs_by_name: Dict[str, NPC] = {n.name: n for n in self.npcs}
        self._audio_by_name: Dict[str, AudioSource] = {a.name: a for a in self.audio_sources}
        self._panels_by_name: Dict[str, KnowledgePanel] = {p.name: p for p in self.knowledge_panels}
        self._last_haptic_sources: set = set()

    # ------------------------------------------------------------------
    # Simulation tick
    # ------------------------------------------------------------------

    def tick(self, user: VRUser) -> List[PlazaEvent]:
        """Advance the plaza simulation by one tick.

        For each tick:

        1. Use :class:`~spinstep.QuaternionDepthIterator` to traverse the
           scene tree and collect nodes whose orientations align with the
           user's gaze.
        2. Check each entity type against the appropriate user cone.
        3. Update entity states (highlight, volume, visibility, NPC attention).
        4. Emit :class:`PlazaEvent` records for everything that changed.

        Args:
            user: The :class:`~vrspin.user.VRUser` whose orientation drives
                all perception this tick.

        Returns:
            List of :class:`PlazaEvent` records produced this tick.
        """
        self.tick_count += 1
        events: List[PlazaEvent] = []

        # -- 1. SpinStep tree traversal: find orientation-aligned nodes ------
        attended_names = set()
        try:
            for node in QuaternionDepthIterator(
                self.root,
                user.orientation,
                angle_threshold=np.deg2rad(45.0),
            ):
                attended_names.add(node.name)
        except Exception:
            pass  # traversal may produce no results; that is fine

        # -- 2. Interactive objects (visual cone) ----------------------------
        for obj in self.objects:
            was_active = obj.active
            in_visual = user.sees(obj.orientation)
            if in_visual and not was_active:
                obj.activate()
                events.append(
                    PlazaEvent(
                        self.tick_count,
                        "visual",
                        obj.name,
                        f"'{obj.name}' enters visual cone — highlighted ✦  {obj.description}",
                    )
                )
            elif not in_visual and was_active:
                obj.deactivate()
                events.append(
                    PlazaEvent(
                        self.tick_count,
                        "visual",
                        obj.name,
                        f"'{obj.name}' leaves visual cone — highlight off",
                    )
                )

        # -- 3. Audio sources (audio cone) -----------------------------------
        for audio in self.audio_sources:
            in_audio = user.hears(audio.orientation)
            was_playing = audio.playing
            if in_audio:
                volume = audio.base_volume
                if not was_playing:
                    audio.start(volume)
                    events.append(
                        PlazaEvent(
                            self.tick_count,
                            "audio",
                            audio.name,
                            f"'{audio.name}' begins — {audio.content} "
                            f"(volume {audio.volume:.2f})",
                        )
                    )
                else:
                    audio.set_volume(volume)
            else:
                attenuated = audio.base_volume * 0.15
                if was_playing and audio.volume > attenuated + 0.01:
                    audio.set_volume(attenuated)
                    events.append(
                        PlazaEvent(
                            self.tick_count,
                            "audio",
                            audio.name,
                            f"'{audio.name}' attenuated to {audio.volume:.2f} "
                            f"(outside audio cone)",
                        )
                    )

        # -- 4. Knowledge panels (visual cone) --------------------------------
        for panel in self.knowledge_panels:
            in_visual = user.sees(panel.orientation)
            if in_visual:
                if not panel.visible:
                    panel.show()
                    page = panel.current_content
                    events.append(
                        PlazaEvent(
                            self.tick_count,
                            "knowledge",
                            panel.name,
                            f"Panel '{panel.name}' appears — "
                            f"{page.heading if page else '?'}: "
                            f"{page.body[:60] if page else ''}…",
                        )
                    )
                else:
                    # Advance page if user keeps rotating toward the panel
                    dist = user.visual_cone.angular_distance_to(panel.orientation)
                    if dist < np.deg2rad(10.0) and panel.advance_page():
                        page = panel.current_content
                        events.append(
                            PlazaEvent(
                                self.tick_count,
                                "knowledge",
                                panel.name,
                                f"Panel '{panel.name}' → page {panel.current_page + 1}: "
                                f"{page.heading if page else '?'}",
                            )
                        )
            else:
                if panel.visible:
                    panel.hide()
                    events.append(
                        PlazaEvent(
                            self.tick_count,
                            "knowledge",
                            panel.name,
                            f"Panel '{panel.name}' fades out",
                        )
                    )

        # -- 5. NPC attention (perception cones) -----------------------------
        for npc in self.npcs:
            npc_events = npc.tick(user)
            for msg in npc_events:
                events.append(
                    PlazaEvent(self.tick_count, "npc", npc.name, msg)
                )

        # -- 6. Haptic feedback (haptic cone) --------------------------------
        haptic_sources = {o.name for o in self.objects if user.feels(o.orientation)}
        new_haptic = haptic_sources - self._last_haptic_sources
        for name in new_haptic:
            events.append(
                PlazaEvent(
                    self.tick_count,
                    "haptic",
                    name,
                    f"Controller pulse — '{name}' is directly ahead",
                )
            )
        self._last_haptic_sources = haptic_sources

        return events

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_object(self, name: str) -> Optional[InteractiveObject]:
        return self._objects_by_name.get(name)

    def get_npc(self, name: str) -> Optional[NPC]:
        return self._npcs_by_name.get(name)

    def get_audio(self, name: str) -> Optional[AudioSource]:
        return self._audio_by_name.get(name)

    def get_panel(self, name: str) -> Optional[KnowledgePanel]:
        return self._panels_by_name.get(name)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _angle_to(q1: np.ndarray, q2: np.ndarray) -> float:
    """Angular distance in radians between two quaternions."""
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    return float(_spinstep_quat_distance(q1, q2))
