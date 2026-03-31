"""Attention manager orchestrator.

The :class:`AttentionManager` maintains a registry of
:class:`~vrspin.models.SceneEntity` objects and evaluates them against an
:class:`~vrspin.cone.AttentionCone` each frame, producing
:class:`~vrspin.models.AttentionResult` lists.
"""

from __future__ import annotations

__all__ = ["AttentionManager"]

from typing import Dict, List

import numpy as np
from numpy.typing import ArrayLike

from .cone import AttentionCone
from .models import AttentionResult, SceneEntity


class AttentionManager:
    """Orchestrator that evaluates registered entities against an attention cone.

    The manager owns a single :class:`~vrspin.cone.AttentionCone` and a
    registry of :class:`~vrspin.models.SceneEntity` objects.  Calling
    :meth:`update` with the user's current quaternion re-points the cone and
    queries all registered entities, returning the attention results.

    Args:
        half_angle_rad: Half-aperture for the internal attention cone.
        label: Modality label forwarded to the cone (default ``"manager"``).

    Attributes:
        cone: The underlying :class:`~vrspin.cone.AttentionCone`.
        registered_entities: Mapping from entity id to
            :class:`~vrspin.models.SceneEntity`.

    Example::

        from vrspin import AttentionManager, SceneEntity
        import numpy as np

        mgr = AttentionManager(half_angle_rad=np.deg2rad(60))
        mgr.register_entity(SceneEntity("obj1", [0, 0, 0], [0, 0, 0, 1], "object"))
        results = mgr.update([0, 0, 0, 1])
    """

    def __init__(
        self,
        half_angle_rad: float = np.deg2rad(60.0),
        label: str = "manager",
    ) -> None:
        self.cone = AttentionCone([0, 0, 0, 1], half_angle_rad, label=label)
        self.registered_entities: Dict[str, SceneEntity] = {}
        self._last_results: List[AttentionResult] = []

    def register_entity(self, entity: SceneEntity) -> None:
        """Add or replace an entity in the managed registry.

        Args:
            entity: The :class:`~vrspin.models.SceneEntity` to register.
        """
        self.registered_entities[entity.id] = entity

    def update(self, user_quaternion: ArrayLike) -> List[AttentionResult]:
        """Re-point the cone and evaluate all registered entities.

        Args:
            user_quaternion: The user's current orientation ``[x, y, z, w]``.

        Returns:
            List of :class:`~vrspin.models.AttentionResult`, one per
            registered entity.
        """
        self.cone.update_orientation(user_quaternion)
        entities = list(self.registered_entities.values())
        self._last_results = self.cone.query_entities(entities)
        return list(self._last_results)

    def get_attended_entities(self) -> List[SceneEntity]:
        """Return entities that were inside the cone on the last update.

        Returns:
            List of :class:`~vrspin.models.SceneEntity` objects whose last
            evaluation yielded ``in_attention == True``.
        """
        attended_ids = {r.entity_id for r in self._last_results if r.in_attention}
        return [e for eid, e in self.registered_entities.items() if eid in attended_ids]
