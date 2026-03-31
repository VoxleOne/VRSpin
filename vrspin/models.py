"""Data models for the VRSpin attention system.

Defines the canonical data contracts specified in the architecture:

- :class:`SceneEntity` — a positioned, oriented object in the scene.
- :class:`AttentionResult` — the output of an attention-cone query against
  a single entity.
"""

from __future__ import annotations

__all__ = ["SceneEntity", "AttentionResult"]

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class SceneEntity:
    """A scene entity with position, orientation, and metadata.

    Attributes:
        id: Unique string identifier.
        position: World-space position ``[x, y, z]``.
        orientation: Orientation quaternion ``[x, y, z, w]``.
        entity_type: Category string (e.g. ``"object"``, ``"audio"``,
            ``"npc"``, ``"panel"``).
        metadata: Arbitrary extra data for the entity.
    """

    id: str
    position: np.ndarray
    orientation: np.ndarray
    entity_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)
        self.orientation = np.asarray(self.orientation, dtype=float)
        norm = np.linalg.norm(self.orientation)
        if norm >= 1e-8:
            self.orientation = self.orientation / norm


@dataclass
class AttentionResult:
    """Result of evaluating a single entity against an attention cone.

    Attributes:
        entity_id: Identifier of the evaluated entity.
        in_attention: ``True`` if the entity falls within the attention cone.
        angular_distance: Geodesic angle in radians between the cone
            direction and the entity orientation.
        weight: Optional relevance score (used in multi-head attention
            merging).
    """

    entity_id: str
    in_attention: bool
    angular_distance: float
    weight: float = 0.0
