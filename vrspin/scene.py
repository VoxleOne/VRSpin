"""Scene entity with position, orientation, and metadata for VR integration.

Provides :class:`SceneEntity` (a SpinStep :class:`~spinstep.Node` subclass),
:class:`AttentionResult`, :class:`AttentionManager`, and the :class:`Observer`
protocol for multi-observer perception.

Any node that has an ``orientation`` and ``attention_cones`` can act as an
observer (the *Multi-Observer* model from SpinStep v0.6.0).
"""

from __future__ import annotations

__all__ = ["SceneEntity", "AttentionResult", "AttentionManager", "Observer"]

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable

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


# ---------------------------------------------------------------------------
# Observer protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Observer(Protocol):
    """Protocol for any node that can act as an observer.

    An observer has an orientation quaternion and a dict of attention cones
    keyed by modality (e.g. ``"visual"``, ``"perception"``).  Any object
    satisfying this protocol can be passed to
    :meth:`AttentionManager.update_observers`.

    Example::

        class MyRobot:
            orientation = np.array([0, 0, 0, 1.0])
            attention_cones = {
                "visual": AttentionCone([0,0,0,1], half_angle=np.radians(45)),
            }
    """

    @property
    def orientation(self) -> np.ndarray: ...

    @property
    def attention_cones(self) -> Dict[str, AttentionCone]: ...


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
        self._last_result: Optional[AttentionResult] = None
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
        self._last_result = AttentionResult(attended=attended, unattended=unattended)
        return self._last_result

    def get_attended_entities(self) -> List[SceneEntity]:
        """Return the entities from the most recent :meth:`update` that were attended.

        Returns:
            List of :class:`SceneEntity` objects that fell inside the
            attention cone on the last :meth:`update` call, sorted by
            descending strength.  Returns an empty list if :meth:`update`
            has not yet been called.
        """
        if self._last_result is None:
            return []
        return [entity for entity, _ in self._last_result.attended]

    # ------------------------------------------------------------------
    # Multi-observer support
    # ------------------------------------------------------------------

    def update_observers(
        self,
        observers: List[Observer],
        cone_half_angle: float,
        falloff: Optional[str] = "linear",
    ) -> Dict[str, AttentionResult]:
        """Query entities from the perspective of multiple observers.

        Each observer independently evaluates all registered entities using
        a temporary :class:`~vrspin.cone.AttentionCone`.

        Args:
            observers: List of objects satisfying the :class:`Observer`
                protocol (must have ``.orientation`` and a ``name`` or
                ``__class__.__name__``).
            cone_half_angle: Half-aperture of the attention cone in radians.
            falloff: Attenuation curve (``'linear'``, ``'cosine'``, or
                ``None``).

        Returns:
            Dict mapping observer names to :class:`AttentionResult` instances.
        """
        results: Dict[str, AttentionResult] = {}
        for observer in observers:
            name = getattr(observer, "name", observer.__class__.__name__)
            result = self.update(observer.orientation, cone_half_angle, falloff)
            results[name] = result
        return results
