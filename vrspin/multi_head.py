"""Multi-head (multi-modal) attention system.

The :class:`MultiHeadAttention` composes three
:class:`~vrspin.cone.AttentionCone` instances — visual, audio, and haptic —
and merges their results using a configurable weighted-sum strategy.
"""

from __future__ import annotations

__all__ = ["MultiHeadAttention"]

from typing import Dict, List

import numpy as np

from .cone import AttentionCone
from .models import AttentionResult, SceneEntity


_MAX_ANGULAR_DISTANCE: float = float(np.pi)


class MultiHeadAttention:
    """Multi-modal attention that merges visual, audio, and haptic cones.

    Each modality cone independently queries the entity list and produces
    :class:`~vrspin.models.AttentionResult` objects.  The
    :meth:`merge_results` method combines per-channel results into a single
    unified list using the configured weights.

    Args:
        visual_cone: :class:`~vrspin.cone.AttentionCone` for visual attention.
        audio_cone: :class:`~vrspin.cone.AttentionCone` for auditory attention.
        haptic_cone: :class:`~vrspin.cone.AttentionCone` for haptic attention.
        weights: Optional mapping of modality name to merge weight.
            Defaults to ``{"visual": 0.7, "audio": 0.2, "haptic": 0.1}``.

    Attributes:
        visual_cone: Visual attention cone.
        audio_cone: Auditory attention cone.
        haptic_cone: Haptic attention cone.
        weights: Per-modality merge weights.

    Example::

        import numpy as np
        from vrspin import AttentionCone, MultiHeadAttention, SceneEntity

        v = AttentionCone([0, 0, 0, 1], np.deg2rad(60), "visual")
        a = AttentionCone([0, 0, 0, 1], np.deg2rad(120), "audio")
        h = AttentionCone([0, 0, 0, 1], np.deg2rad(30), "haptic")
        mha = MultiHeadAttention(v, a, h)
        entity = SceneEntity("obj", [0, 0, 0], [0, 0, 0, 1], "object")
        results = mha.update([entity])
    """

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "visual": 0.7,
        "audio": 0.2,
        "haptic": 0.1,
    }

    def __init__(
        self,
        visual_cone: AttentionCone,
        audio_cone: AttentionCone,
        haptic_cone: AttentionCone,
        weights: Dict[str, float] | None = None,
    ) -> None:
        self.visual_cone = visual_cone
        self.audio_cone = audio_cone
        self.haptic_cone = haptic_cone
        self.weights: Dict[str, float] = dict(weights or self.DEFAULT_WEIGHTS)

    def update(self, entities: List[SceneEntity]) -> List[AttentionResult]:
        """Query all three cones and merge the results.

        Args:
            entities: Scene entities to evaluate.

        Returns:
            Merged list of :class:`~vrspin.models.AttentionResult`, one per
            entity.
        """
        results_by_channel: Dict[str, List[AttentionResult]] = {
            "visual": self.visual_cone.query_entities(entities),
            "audio": self.audio_cone.query_entities(entities),
            "haptic": self.haptic_cone.query_entities(entities),
        }
        return self.merge_results(results_by_channel)

    def merge_results(
        self,
        results_by_channel: Dict[str, List[AttentionResult]],
    ) -> List[AttentionResult]:
        """Merge per-channel results using a weighted sum.

        For each entity the merged weight is::

            weight = Σ (channel_weight × (1 − angular_distance / π))

        summed over channels where the entity is ``in_attention``.  An entity
        is considered attended in the merged output when **any** channel
        reports it as in-attention.

        Args:
            results_by_channel: Mapping of channel name to per-entity results.
                All lists must be the same length and in the same entity order.

        Returns:
            Merged list of :class:`~vrspin.models.AttentionResult`.
        """
        # Collect entity ids from the first non-empty channel
        first_channel = next(iter(results_by_channel.values()), [])
        entity_ids = [r.entity_id for r in first_channel]

        merged: List[AttentionResult] = []
        for idx, eid in enumerate(entity_ids):
            total_weight = 0.0
            any_in_attention = False
            min_distance = _MAX_ANGULAR_DISTANCE

            for channel_name, channel_results in results_by_channel.items():
                r = channel_results[idx]
                channel_w = self.weights.get(channel_name, 0.0)
                if r.in_attention:
                    any_in_attention = True
                    proximity = 1.0 - r.angular_distance / _MAX_ANGULAR_DISTANCE
                    total_weight += channel_w * proximity
                min_distance = min(min_distance, r.angular_distance)

            merged.append(
                AttentionResult(
                    entity_id=eid,
                    in_attention=any_in_attention,
                    angular_distance=min_distance,
                    weight=total_weight,
                )
            )
        return merged
