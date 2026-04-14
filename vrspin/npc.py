"""NPC with SpinStep-powered directional perception.

Each NPC owns a SpinStep :class:`~spinstep.Node` for scene-tree placement and
an :class:`~vrspin.cone.AttentionCone` for perception.  The NPC notices users
that enter its cone and smoothly rotates toward them via quaternion SLERP.
"""

from __future__ import annotations

__all__ = ["NPC", "NPCState", "NPCAttentionAgent"]

from enum import Enum, auto
from typing import List, Optional

import numpy as np
from numpy.typing import ArrayLike

from spinstep import Node
from spinstep.utils import (
    is_within_angle_threshold,
    quaternion_distance,
    forward_vector_from_quaternion,
    direction_to_quaternion,
)

from .cone import AttentionCone
from .user import VRUser
from .utils import slerp as _vrspin_slerp


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
        new_quat = _vrspin_slerp(
            self.node.orientation, self.target_orientation, self.slerp_speed
        )
        new_quat = new_quat / np.linalg.norm(new_quat)
        # Update node orientation (replace array values in place)
        self.node.orientation[:] = new_quat
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
                # Target: face back toward the user (quaternion conjugate)
                # For unit quaternions, inv() == conjugate: [-x, -y, -z, w]
                uq = user.orientation
                facing_user = np.array([-uq[0], -uq[1], -uq[2], uq[3]])
                self._set_target_orientation(facing_user)

            if self.state in (NPCState.NOTICING, NPCState.ENGAGED):
                self._step_slerp()

            # Check whether fully turned
            if self.target_orientation is not None:
                angle_remaining = quaternion_distance(
                    self.node.orientation, self.target_orientation
                )
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


# ---------------------------------------------------------------------------
# NPCAttentionAgent — lightweight agent from the instruction-style API
# ---------------------------------------------------------------------------


class NPCAttentionAgent:
    """An NPC that perceives and reacts to entities within its attention cone.

    The NPC has its own perception cone.  When a user enters the cone, the
    NPC smoothly rotates toward them via quaternion SLERP.  When the user
    leaves, the NPC returns to its idle orientation.

    This class operates on raw quaternions and does not require a
    :class:`~vrspin.user.VRUser` instance, making it suitable for
    the engine-agnostic scene API (:mod:`vrspin.scene`).

    Args:
        entity: A scene entity (anything with ``.name`` and ``.orientation``
            attributes) representing this NPC.
        perception_half_angle: Half-angle of the NPC's perception cone
            in radians.
        turn_speed: Interpolation factor for smooth turning (``0``–``1`` per
            update).  Higher values mean faster turning.
        idle_orientation: Default orientation when no target is attended.
            If ``None``, the entity's initial orientation is used.

    Example::

        from vrspin.scene import SceneEntity
        from vrspin.npc import NPCAttentionAgent
        import numpy as np

        npc_entity = SceneEntity("vendor", [0, 0, 0, 1], entity_type="npc")
        npc = NPCAttentionAgent(npc_entity, perception_half_angle=np.radians(40))
        user_quat = [0.1, 0, 0, 0.995]
        if npc.is_aware_of(user_quat):
            npc.update(targets=[user_quat], dt=1 / 60)
    """

    def __init__(
        self,
        entity: object,
        perception_half_angle: float,
        turn_speed: float = 0.1,
        idle_orientation: Optional[ArrayLike] = None,
    ) -> None:
        self.entity = entity
        self.perception_half_angle: float = float(perception_half_angle)
        self.turn_speed: float = float(turn_speed)
        self.idle_orientation: np.ndarray = (
            np.array(idle_orientation, dtype=float)
            if idle_orientation is not None
            else np.array(entity.orientation, dtype=float).copy()
        )

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------

    def is_aware_of(self, target_quat: ArrayLike) -> bool:
        """Return ``True`` if *target_quat* is within the perception cone.

        Delegates to SpinStep's :func:`~spinstep.utils.is_within_angle_threshold`.

        Args:
            target_quat: Quaternion ``[x, y, z, w]`` to test.
        """
        target = np.asarray(target_quat, dtype=float)
        norm = np.linalg.norm(target)
        if norm < 1e-8:
            return False
        target = target / norm
        return bool(
            is_within_angle_threshold(
                self.entity.orientation, target, self.perception_half_angle
            )
        )

    # ------------------------------------------------------------------
    # Smooth rotation
    # ------------------------------------------------------------------

    def face_toward(self, target_quat: ArrayLike, t: float) -> None:
        """SLERP the entity's orientation toward *target_quat* by fraction *t*.

        Args:
            target_quat: Desired orientation ``[x, y, z, w]``.
            t: Interpolation fraction in ``[0, 1]``.
        """
        target = np.asarray(target_quat, dtype=float)
        target = target / np.linalg.norm(target)

        new_quat = _vrspin_slerp(self.entity.orientation, target, t)
        new_quat = new_quat / np.linalg.norm(new_quat)
        self.entity.orientation[:] = new_quat

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(self, targets: List[ArrayLike], dt: float) -> None:
        """Advance the NPC by one simulation step.

        Finds the closest target inside the perception cone and SLERP-rotates
        toward it.  If no targets are in range the NPC rotates back toward
        its idle orientation.

        Args:
            targets: List of quaternions ``[x, y, z, w]`` representing
                potential attention targets (e.g. users).
            dt: Time-step in seconds (used to scale the turn speed).
        """
        # Find closest target within cone
        best_target = None
        best_dist = float("inf")
        for tq in targets:
            tq_arr = np.asarray(tq, dtype=float)
            norm = np.linalg.norm(tq_arr)
            if norm < 1e-8:
                continue
            tq_arr = tq_arr / norm
            dist = float(quaternion_distance(self.entity.orientation, tq_arr))
            if dist < self.perception_half_angle and dist < best_dist:
                best_dist = dist
                best_target = tq_arr

        t = min(1.0, self.turn_speed * dt * 60.0)  # normalize for ~60 fps
        if best_target is not None:
            self.face_toward(best_target, t)
        else:
            # Return to idle orientation
            self.face_toward(self.idle_orientation, t)

