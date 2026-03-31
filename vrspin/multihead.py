"""Multi-modal attention with independent cones per modality.

Provides :class:`MultiHeadAttention`, which maintains multiple
:class:`~vrspin.cone.AttentionCone` instances — one per sensory modality —
and queries them all in a single :meth:`update` call.
"""

from __future__ import annotations

__all__ = ["MultiHeadAttention"]

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from .cone import AttentionCone
from .scene import SceneEntity


class MultiHeadAttention:
    """Multiple independent attention cones for different modalities.

    Each *head* is an :class:`~vrspin.cone.AttentionCone` with its own
    half-angle and falloff, representing a different sensory channel
    (visual, audio, haptic, etc.).

    Args:
        heads: Dict mapping modality names to
            :class:`~vrspin.cone.AttentionCone` instances.

    Example::

        from vrspin import AttentionCone
        from vrspin.multihead import MultiHeadAttention
        import numpy as np

        user_quat = [0, 0, 0, 1]
        multi = MultiHeadAttention({
            'visual': AttentionCone(user_quat, half_angle=np.radians(45)),
            'audio':  AttentionCone(user_quat, half_angle=np.radians(90)),
            'haptic': AttentionCone(user_quat, half_angle=np.radians(20)),
        })
        results = multi.update(user_quat, entities)
        # results['visual'] → [(entity, strength), ...]
    """

    def __init__(self, heads: Dict[str, AttentionCone]) -> None:
        self.heads: Dict[str, AttentionCone] = dict(heads)
        self._last_results: Dict[str, List[Tuple[SceneEntity, float]]] = {}

    def update(
        self,
        origin_quat: ArrayLike,
        entities: List[SceneEntity],
    ) -> Dict[str, List[Tuple[SceneEntity, float]]]:
        """Query all modalities and return per-head results.

        Each head's origin orientation is updated to *origin_quat* before
        querying, so that all cones point in the same direction (the user's
        current head pose).

        Args:
            origin_quat: User orientation ``[x, y, z, w]`` to apply to all
                heads.
            entities: Scene entities to evaluate.

        Returns:
            Dict mapping modality names to lists of
            ``(SceneEntity, attenuation)`` tuples for entities inside that
            head's cone.  Each list is sorted by descending attenuation.
        """
        results: Dict[str, List[Tuple[SceneEntity, float]]] = {}
        for name, cone in self.heads.items():
            cone.update_origin(origin_quat)
            attended: List[Tuple[SceneEntity, float]] = []
            for entity in entities:
                strength = cone.attenuation(entity.orientation)
                if strength > 0.0:
                    attended.append((entity, strength))
            attended.sort(key=lambda t: t[1], reverse=True)
            results[name] = attended
        self._last_results = results
        return results

    def merge_results(
        self,
        strategy: str = "union",
    ) -> List[SceneEntity]:
        """Merge the most recent per-head results into a single entity list.

        Args:
            strategy: Merge strategy.  ``'union'`` returns entities attended
                by *any* head.  ``'intersection'`` returns entities attended
                by *all* heads.

        Returns:
            De-duplicated list of :class:`~vrspin.scene.SceneEntity` objects.

        Raises:
            ValueError: If *strategy* is not recognised.
        """
        if strategy not in ("union", "intersection"):
            raise ValueError(f"Unknown merge strategy {strategy!r}")

        if not self._last_results:
            return []

        sets = [
            {id(entity) for entity, _ in items}
            for items in self._last_results.values()
        ]
        entity_map = {
            id(entity): entity
            for items in self._last_results.values()
            for entity, _ in items
        }

        if strategy == "union":
            merged_ids = set().union(*sets) if sets else set()
        else:  # intersection
            merged_ids = set.intersection(*sets) if sets else set()

        return [entity_map[eid] for eid in merged_ids if eid in entity_map]
