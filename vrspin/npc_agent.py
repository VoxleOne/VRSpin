"""NPC attention agent with awareness tracking.

The :class:`NPCAttentionAgent` models an NPC's perception as an
:class:`~vrspin.cone.AttentionCone` combined with a temporal awareness
list.  It tracks which entities have been recently observed, their
confidence levels, and can compute rotations toward a target.
"""

from __future__ import annotations

__all__ = ["NPCAttentionAgent"]

import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as R

from .cone import AttentionCone
from .models import SceneEntity


@dataclass
class _AwarenessEntry:
    """Internal record of an entity the NPC is aware of."""

    entity_id: str
    last_seen_timestamp: float
    confidence: float


class NPCAttentionAgent:
    """An NPC perception agent with awareness tracking.

    The agent owns an :class:`~vrspin.cone.AttentionCone` and maintains a
    list of entities it is aware of, along with timestamps and confidence
    scores.

    Args:
        orientation: Initial quaternion ``[x, y, z, w]``.
        perception_half_angle: Cone half-angle in radians.
        confidence_decay: Per-update multiplicative decay applied to
            entities **not** currently in the cone.

    Attributes:
        orientation: Current NPC orientation.
        cone: Underlying :class:`~vrspin.cone.AttentionCone`.
        awareness: Mapping from entity id to awareness data.

    Example::

        import numpy as np
        from vrspin import NPCAttentionAgent, SceneEntity

        agent = NPCAttentionAgent([0, 0, 0, 1])
        entity = SceneEntity("player", [0, 0, 0], [0, 0, 0, 1], "user")
        agent.update([0, 0, 0, 1], [entity])
        print(agent.is_aware_of("player"))  # True
    """

    def __init__(
        self,
        orientation: ArrayLike,
        perception_half_angle: float = np.deg2rad(120.0),
        confidence_decay: float = 0.9,
    ) -> None:
        arr = np.asarray(orientation, dtype=float)
        norm = np.linalg.norm(arr)
        if norm < 1e-8:
            raise ValueError("Orientation quaternion must be non-zero")
        self.orientation: np.ndarray = arr / norm
        self.cone = AttentionCone(self.orientation, perception_half_angle, label="npc_agent")
        self._confidence_decay = float(confidence_decay)
        self.awareness: Dict[str, _AwarenessEntry] = {}

    def update(
        self,
        user_quaternion: ArrayLike,
        entities: List[SceneEntity],
    ) -> None:
        """Update the agent's awareness based on the current user pose.

        Entities inside the perception cone have their confidence set to
        ``1.0``; entities outside the cone have their confidence decayed.

        Args:
            user_quaternion: Current user orientation ``[x, y, z, w]``
                (used to re-orient the agent's perception cone).
            entities: Scene entities to evaluate.
        """
        self.cone.update_orientation(user_quaternion)
        self.orientation = self.cone.orientation.copy()
        now = time.monotonic()

        observed_ids: set = set()
        for entity in entities:
            dist = self.cone.angular_distance_to(entity.orientation)
            in_cone = dist < self.cone.half_angle
            if in_cone:
                observed_ids.add(entity.id)
                self.awareness[entity.id] = _AwarenessEntry(
                    entity_id=entity.id,
                    last_seen_timestamp=now,
                    confidence=1.0,
                )

        # Decay confidence for entities not currently observed
        for eid in list(self.awareness):
            if eid not in observed_ids:
                entry = self.awareness[eid]
                entry.confidence *= self._confidence_decay
                if entry.confidence < 0.01:
                    del self.awareness[eid]

    def is_aware_of(self, entity_id: str) -> bool:
        """Return ``True`` if the agent currently has awareness of *entity_id*.

        An entity is considered "aware" when its confidence is above the
        decay threshold (i.e. it has not fully decayed away).

        Args:
            entity_id: The id of the entity to check.
        """
        return entity_id in self.awareness

    def compute_rotation_to_target(self, target_quaternion: ArrayLike) -> np.ndarray:
        """Compute the delta rotation needed to face *target_quaternion*.

        Args:
            target_quaternion: Desired facing orientation ``[x, y, z, w]``.

        Returns:
            Delta quaternion ``[x, y, z, w]`` that, when composed with the
            agent's current orientation, produces the target orientation.
        """
        target = np.asarray(target_quaternion, dtype=float)
        norm = np.linalg.norm(target)
        if norm < 1e-8:
            return np.array([0.0, 0.0, 0.0, 1.0])
        target = target / norm
        delta = R.from_quat(self.orientation).inv() * R.from_quat(target)
        return delta.as_quat()
