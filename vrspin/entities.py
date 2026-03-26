"""Scene entities: interactive objects, spatial audio sources, and knowledge panels.

Every entity wraps a SpinStep :class:`~spinstep.Node` so that it can
participate in quaternion-driven scene-tree traversal.  The node's orientation
encodes the entity's *direction from the plaza centre*, allowing
:class:`~spinstep.QuaternionDepthIterator` to naturally visit entities that
align with the user's current gaze.
"""

from __future__ import annotations

__all__ = ["InteractiveObject", "AudioSource", "KnowledgePanel"]

from dataclasses import dataclass, field
from typing import List

import numpy as np
from numpy.typing import ArrayLike

from spinstep import Node


# ---------------------------------------------------------------------------
# InteractiveObject
# ---------------------------------------------------------------------------


class InteractiveObject:
    """A visible, interactive prop in the virtual plaza.

    Objects highlight and become "active" when they fall inside the user's
    visual attention cone.

    Args:
        name: Display name (e.g. ``"Fountain"``).
        orientation: Quaternion ``[x, y, z, w]`` — direction from plaza centre.
        description: Short flavour text shown when activated.

    Attributes:
        node: Underlying SpinStep :class:`~spinstep.Node`.
        highlighted: Whether the object is currently glowing.
        active: Whether the object has been fully activated this tick.
        description: Flavour text.

    Example::

        from vrspin import InteractiveObject
        import numpy as np

        fountain = InteractiveObject("Fountain", [0, 0, 0, 1], "A beautiful stone fountain")
    """

    def __init__(
        self,
        name: str,
        orientation: ArrayLike,
        description: str = "",
    ) -> None:
        self.node: Node = Node(name, orientation)
        self.highlighted: bool = False
        self.active: bool = False
        self.description: str = description

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def orientation(self) -> np.ndarray:
        return self.node.orientation

    def activate(self) -> None:
        """Mark object as highlighted and active."""
        self.highlighted = True
        self.active = True

    def deactivate(self) -> None:
        """Remove highlight and active state."""
        self.highlighted = False
        self.active = False

    def __repr__(self) -> str:
        state = "highlighted" if self.highlighted else "idle"
        return f"InteractiveObject({self.name!r}, {state})"


# ---------------------------------------------------------------------------
# AudioSource
# ---------------------------------------------------------------------------


class AudioSource:
    """A spatial audio emitter in the virtual plaza.

    Volume is boosted when the source falls inside the user's audio cone
    and attenuated otherwise, simulating directional hearing.

    Args:
        name: Display name (e.g. ``"FountainAmbience"``).
        orientation: Quaternion ``[x, y, z, w]`` — direction from plaza centre.
        content: Description of the sound being emitted.
        base_volume: Baseline volume in the range ``[0, 1]`` when audible.

    Attributes:
        node: Underlying SpinStep :class:`~spinstep.Node`.
        volume: Current perceived volume ``[0, 1]``.
        playing: Whether the source is currently playing.
        content: Description of the audio.
        base_volume: Baseline volume.

    Example::

        from vrspin import AudioSource

        music = AudioSource("MarketMusic", [0, 0.707, 0, 0.707], "Lively merchant tunes")
    """

    def __init__(
        self,
        name: str,
        orientation: ArrayLike,
        content: str = "",
        base_volume: float = 0.8,
    ) -> None:
        self.node: Node = Node(name, orientation)
        self.volume: float = 0.0
        self.playing: bool = False
        self.content: str = content
        self.base_volume: float = float(base_volume)

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def orientation(self) -> np.ndarray:
        return self.node.orientation

    def start(self, volume: float | None = None) -> None:
        """Begin playback at *volume* (defaults to :attr:`base_volume`)."""
        self.playing = True
        self.volume = volume if volume is not None else self.base_volume

    def stop(self) -> None:
        """Stop playback and mute the source."""
        self.playing = False
        self.volume = 0.0

    def set_volume(self, volume: float) -> None:
        """Set the current volume, clamped to ``[0, 1]``."""
        self.volume = max(0.0, min(1.0, volume))

    def __repr__(self) -> str:
        state = f"playing @ {self.volume:.2f}" if self.playing else "silent"
        return f"AudioSource({self.name!r}, {state})"


# ---------------------------------------------------------------------------
# KnowledgePanel
# ---------------------------------------------------------------------------


@dataclass
class PanelPage:
    """A single page of content in a :class:`KnowledgePanel`.

    Args:
        heading: Short title displayed at the top.
        body: Main text body.
        trigger_angle_deg: Angle offset (degrees) from the panel's base
            orientation at which this page is shown.  Page 0 is shown when
            the user first looks at the panel; subsequent pages appear as the
            user continues rotating in the same direction.
    """

    heading: str
    body: str
    trigger_angle_deg: float = 0.0


class KnowledgePanel:
    """A floating information panel that reveals content as the user rotates.

    The panel becomes visible when the user's gaze enters its orientation cone.
    As the user continues rotating past the panel, the content pages advance,
    simulating *directional embeddings* of knowledge in space.

    Args:
        name: Panel identifier.
        orientation: Quaternion ``[x, y, z, w]`` — direction from plaza centre.
        pages: List of :class:`PanelPage` items, ordered by trigger angle.

    Attributes:
        node: Underlying SpinStep :class:`~spinstep.Node`.
        visible: Whether the panel is currently shown to the user.
        current_page: Index of the currently displayed page.
        pages: Ordered list of :class:`PanelPage` content.

    Example::

        from vrspin.entities import KnowledgePanel, PanelPage

        panel = KnowledgePanel(
            "VR Art",
            [0, 0.259, 0, 0.966],
            pages=[
                PanelPage("VR Art Intro", "Creating immersive art in virtual reality..."),
                PanelPage("Interaction Design", "How orientation shapes UX..."),
            ],
        )
    """

    def __init__(
        self,
        name: str,
        orientation: ArrayLike,
        pages: List[PanelPage] | None = None,
    ) -> None:
        self.node: Node = Node(name, orientation)
        self.visible: bool = False
        self.current_page: int = 0
        self.pages: List[PanelPage] = pages or []

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def orientation(self) -> np.ndarray:
        return self.node.orientation

    def show(self) -> None:
        """Make the panel visible, resetting to page 0."""
        self.visible = True
        self.current_page = 0

    def hide(self) -> None:
        """Hide the panel."""
        self.visible = False

    def advance_page(self) -> bool:
        """Move to the next content page if available.

        Returns:
            ``True`` when a new page is now showing, ``False`` if already
            at the last page.
        """
        if self.current_page < len(self.pages) - 1:
            self.current_page += 1
            return True
        return False

    @property
    def current_content(self) -> PanelPage | None:
        """The currently displayed :class:`PanelPage`, or ``None`` if empty."""
        if not self.pages:
            return None
        return self.pages[self.current_page]

    def __repr__(self) -> str:
        state = f"visible page={self.current_page}" if self.visible else "hidden"
        return f"KnowledgePanel({self.name!r}, {state})"
