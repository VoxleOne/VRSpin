"""NPC with SpinStep-powered directional perception.

Each NPC owns a SpinStep :class:`~spinstep.Node` for scene-tree placement and
an :class:`~vrspin.cone.AttentionCone` for perception.  The NPC notices users
that enter its cone and smoothly rotates toward them via quaternion SLERP.
"""

from __future__ import annotations

__all__ = ["NPC", "NPCState"]

from enum import Enum, auto
from typing import List, Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as R, Slerp

from spinstep import Node

from .cone import AttentionCone
from .user import VRUser


class NPCState(Enum):
    """Observable states of an NPC's attention machine."""

    IDLE = auto()        # standing still, ignoring environment
    NOTICING = auto()    # a user has entered the cone; turning toward them
    ENGAGED = auto()     # fully facing the user, ready to interact
    SPEAKING = auto()    # playing a dialogue line


class NPC:
    """A non-player character whose perception is orientation-driven.

    The NPC uses a SpinStep :class:`~spinstep.Node` for scene-graph
    participation and an :class:`~vrspin.cone.AttentionCone` (default 120°
    half-angle) to detect when a :class:`~vrspin.user.VRUser` enters its
    field of view.  Once noticed, it SLERP-rotates toward the user over
    multiple :meth:`tick` calls.

    Args:
        name: Display name (e.g. ``"Elena"``).
        orientation: Initial quaternion ``[x, y, z, w]`` orientation.
        perception_half_angle: Cone half-angle in radians (default 120°).
        slerp_speed: Fraction ``[0, 1]`` of the remaining rotation applied
            per :meth:`tick`; larger values produce faster turning.
        greeting: Dialogue line spoken when fully engaged.

    Attributes:
        node: Underlying SpinStep :class:`~spinstep.Node`.
        perception_cone: :class:`~vrspin.cone.AttentionCone`.
        state: Current :class:`NPCState`.
        target_orientation: Orientation the NPC is rotating toward, or
            ``None`` when idle.
        greeting: Dialogue string.
        noticed_users: List of currently noticed user names.

    Example::

        import numpy as np
        from vrspin import NPC, VRUser

        npc = NPC("Elena", [0, 0, 0, 1])
        user = VRUser("Alice")
        events = npc.tick(user)
        for ev in events:
            print(ev)
    """

    DEFAULT_PERCEPTION_HALF_ANGLE: float = np.deg2rad(120.0)
    ENGAGED_THRESHOLD: float = np.deg2rad(5.0)

    def __init__(
        self,
        name: str,
        orientation: ArrayLike,
        perception_half_angle: float = DEFAULT_PERCEPTION_HALF_ANGLE,
        slerp_speed: float = 0.15,
        greeting: str = "",
    ) -> None:
        self.node: Node = Node(name, orientation)
        self.perception_cone = AttentionCone(
            orientation, perception_half_angle, label="npc_perception"
        )
        self.state: NPCState = NPCState.IDLE
        self.target_orientation: Optional[np.ndarray] = None
        self.slerp_speed: float = float(slerp_speed)
        self.greeting: str = greeting or f"Hello, I'm {name}."
        self.noticed_users: List[str] = []
        self._greeted_users: List[str] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def orientation(self) -> np.ndarray:
        return self.node.orientation

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------

    def user_in_cone(self, user: VRUser) -> bool:
        """Return ``True`` if the user's orientation is within perception cone.

        The check uses the *user's orientation* quaternion as a proxy for the
        user's direction relative to the NPC, enabling a pure
        orientation-space attention model.

        Args:
            user: The :class:`~vrspin.user.VRUser` to test.
        """
        return self.perception_cone.is_in_cone(user.orientation)

    # ------------------------------------------------------------------
    # Rotation toward target
    # ------------------------------------------------------------------

    def _set_target_orientation(self, target_quat: np.ndarray) -> None:
        """Set a new target orientation to SLERP toward."""
        self.target_orientation = target_quat / np.linalg.norm(target_quat)

    def _step_slerp(self) -> None:
        """Advance one SLERP step toward :attr:`target_orientation`."""
        if self.target_orientation is None:
            return
        current = R.from_quat(self.node.orientation)
        target = R.from_quat(self.target_orientation)
        times = [0.0, 1.0]
        key_rots = R.from_quat(
            np.stack([current.as_quat(), target.as_quat()])
        )
        slerp = Slerp(times, key_rots)
        new_rot = slerp([self.slerp_speed])[0]
        new_quat = new_rot.as_quat()
        # Update node orientation (replace array values in place)
        self.node.orientation[:] = new_quat / np.linalg.norm(new_quat)
        self.perception_cone.update_orientation(self.node.orientation)

    # ------------------------------------------------------------------
    # Simulation tick
    # ------------------------------------------------------------------

    def tick(self, user: VRUser) -> List[str]:
        """Advance the NPC's attention state machine by one simulation step.

        1. Check whether the user has entered / left the perception cone.
        2. Determine the orientation that would make the NPC "face" the user
           (approximated as the inverse of the user's orientation — the NPC
           turns to "look back").
        3. SLERP toward that orientation.
        4. Transition state and emit event strings.

        Args:
            user: The :class:`~vrspin.user.VRUser` to react to.

        Returns:
            List of human-readable event strings describing what changed.
        """
        events: List[str] = []
        in_cone = self.user_in_cone(user)

        if in_cone:
            if user.name not in self.noticed_users:
                self.noticed_users.append(user.name)
                self.state = NPCState.NOTICING
                events.append(
                    f"NPC {self.name!r} notices {user.name!r} — begins rotating"
                )
                # Target: face back toward the user (inverse of user orientation)
                facing_user = R.from_quat(user.orientation).inv().as_quat()
                self._set_target_orientation(facing_user)

            if self.state in (NPCState.NOTICING, NPCState.ENGAGED):
                self._step_slerp()

            # Check whether fully turned
            if self.target_orientation is not None:
                angle_remaining = (
                    R.from_quat(self.node.orientation).inv()
                    * R.from_quat(self.target_orientation)
                ).magnitude()
                if angle_remaining < self.ENGAGED_THRESHOLD and self.state == NPCState.NOTICING:
                    self.state = NPCState.ENGAGED
                    events.append(
                        f"NPC {self.name!r} is now fully facing {user.name!r}"
                    )
                    if user.name not in self._greeted_users:
                        self._greeted_users.append(user.name)
                        self.state = NPCState.SPEAKING
                        events.append(
                            f"NPC {self.name!r} says: \"{self.greeting}\""
                        )
        else:
            # User left the cone
            if user.name in self.noticed_users:
                self.noticed_users.remove(user.name)
                if user.name in self._greeted_users:
                    self._greeted_users.remove(user.name)
                self.state = NPCState.IDLE
                self.target_orientation = None
                events.append(
                    f"NPC {self.name!r} loses sight of {user.name!r} — returning to idle"
                )

        return events

    def __repr__(self) -> str:
        return f"NPC({self.name!r}, state={self.state.name})"
