"""Scene entity with position, orientation, and metadata for VR integration.

Provides :class:`SceneEntity` (a SpinStep :class:`~spinstep.Node` subclass),
:class:`AttentionResult`, and :class:`AttentionManager` for querying which
entities fall within the user's attention cone each frame.
"""

from __future__ import annotations

__all__ = ["SceneEntity", "AttentionResult", "AttentionManager"]

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from spinstep import Node

from .cone import AttentionCone


class SceneEntity(Node):
    """A scene-graph node with 3D position and VR-specific metadata.

    Extends :class:`~spinstep.Node` with a world-space position, entity type,
    and arbitrary metadata needed for VR scene management.

    Args:
        name: Entity identifier.
        orientation: Quaternion ``[x, y, z, w]``.
        position: 3D world position ``[x, y, z]``.  Defaults to the origin.
        entity_type: One of ``'npc'``, ``'object'``, ``'panel'``,
            ``'audio_source'``.
        metadata: Arbitrary key-value data for the entity.

    Attributes:
        position: World-space position as a NumPy array of shape ``(3,)``.
        entity_type: Entity category string.
        metadata: Extra data dictionary.

    Example::

        from vrspin.scene import SceneEntity
        from spinstep.utils import quaternion_from_euler

        fountain = SceneEntity(
            name="fountain",
            orientation=quaternion_from_euler([0, 0, 0]),
            position=[5.0, 0.0, 3.0],
            entity_type="object",
        )
    """

    def __init__(
        self,
        name: str,
        orientation: ArrayLike,
        position: ArrayLike = (0.0, 0.0, 0.0),
        entity_type: str = "object",
        metadata: Optional[Dict] = None,
    ) -> None:
        super().__init__(name, orientation)
        self.position: np.ndarray = np.asarray(position, dtype=float)
        self.entity_type: str = entity_type
        self.metadata: Dict = metadata or {}

    @property
    def direction_quaternion(self) -> np.ndarray:
        """Return the orientation quaternion of this entity.

        Returns:
            NumPy array of shape ``(4,)`` — ``[x, y, z, w]``.
        """
        return self.orientation

    def distance_to(self, other: SceneEntity) -> float:
        """Euclidean distance to another :class:`SceneEntity`.

        Args:
            other: Another scene entity.

        Returns:
            Straight-line distance in world units.
        """
        return float(np.linalg.norm(self.position - other.position))

    def __repr__(self) -> str:
        return (
            f"SceneEntity({self.name!r}, type={self.entity_type!r}, "
            f"pos={self.position.tolist()})"
        )


@dataclass
class AttentionResult:
    """Result of an attention query — attended entities with attenuation values.

    Attributes:
        attended: List of ``(SceneEntity, float)`` tuples for entities inside
            the cone, sorted by descending attenuation.
        unattended: Entities outside the cone.
    """

    attended: List[Tuple[SceneEntity, float]] = field(default_factory=list)
    unattended: List[SceneEntity] = field(default_factory=list)


class AttentionManager:
    """Manages attention queries across all scene entities.

    Maintains a registry of :class:`SceneEntity` instances and efficiently
    queries which entities fall within an attention cone each frame.

    Args:
        entities: Initial list of scene entities.

    Example::

        from vrspin.scene import SceneEntity, AttentionManager
        import numpy as np

        manager = AttentionManager(entities)
        result = manager.update(user_head_quaternion, cone_half_angle=0.5)
        for entity, strength in result.attended:
            print(f"{entity.name}: {strength:.2f}")
    """

    def __init__(self, entities: Optional[List[SceneEntity]] = None) -> None:
        self._entities: Dict[str, SceneEntity] = {}
        for ent in (entities or []):
            self.register_entity(ent)

    @property
    def entities(self) -> List[SceneEntity]:
        """All registered entities."""
        return list(self._entities.values())

    def register_entity(self, entity: SceneEntity) -> None:
        """Add an entity to the manager.

        Args:
            entity: The :class:`SceneEntity` to register.
        """
        self._entities[entity.name] = entity

    def unregister_entity(self, name: str) -> None:
        """Remove an entity by name.

        Args:
            name: Entity identifier to remove.

        Raises:
            KeyError: If the entity is not found.
        """
        del self._entities[name]

    def update(
        self,
        user_quat: ArrayLike,
        cone_half_angle: float,
        falloff: Optional[str] = "linear",
    ) -> AttentionResult:
        """Query all registered entities against an attention cone.

        Creates a temporary :class:`~vrspin.cone.AttentionCone` from
        *user_quat* and *cone_half_angle*, then classifies every entity as
        attended or unattended.

        Args:
            user_quat: User head orientation ``[x, y, z, w]``.
            cone_half_angle: Half-aperture of the attention cone in radians.
            falloff: Attenuation curve (``'linear'``, ``'cosine'``, or
                ``None``).

        Returns:
            An :class:`AttentionResult` with attended entities sorted by
            descending strength.
        """
        cone = AttentionCone(
            user_quat, half_angle_rad=cone_half_angle, falloff=falloff,
        )
        attended: List[Tuple[SceneEntity, float]] = []
        unattended: List[SceneEntity] = []
        for entity in self._entities.values():
            strength = cone.attenuation(entity.orientation)
            if strength > 0.0:
                attended.append((entity, strength))
            else:
                unattended.append(entity)
        attended.sort(key=lambda t: t[1], reverse=True)
        return AttentionResult(attended=attended, unattended=unattended)
