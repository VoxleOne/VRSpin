"""Bridge protocol message types for engine ↔ Python communication.

Defines the JSON-serialisable message dataclasses for the WebSocket bridge
described in the architecture specification.  The bridge is engine-agnostic:
it works with Unity3D, Unreal Engine, or any other runtime that speaks
JSON over WebSockets.
"""

from __future__ import annotations

__all__ = [
    "UpdateUserPose",
    "UpdateSceneEntities",
    "AttentionResultsMessage",
]

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from .models import AttentionResult, SceneEntity


@dataclass
class UpdateUserPose:
    """Engine → Python: new user head pose.

    Attributes:
        position: World-space position ``[x, y, z]``.
        orientation: Head orientation quaternion ``[x, y, z, w]``.
    """

    position: List[float]
    orientation: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "update_user_pose",
            "payload": {
                "position": list(self.position),
                "orientation": list(self.orientation),
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UpdateUserPose":
        payload = data["payload"]
        return cls(
            position=list(payload["position"]),
            orientation=list(payload["orientation"]),
        )


@dataclass
class UpdateSceneEntities:
    """Engine → Python: updated set of scene entities.

    Attributes:
        entities: List of :class:`~vrspin.models.SceneEntity` objects.
    """

    entities: List[SceneEntity]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "update_scene_entities",
            "payload": {
                "entities": [
                    {
                        "id": e.id,
                        "position": e.position.tolist(),
                        "orientation": e.orientation.tolist(),
                        "entity_type": e.entity_type,
                        "metadata": e.metadata,
                    }
                    for e in self.entities
                ],
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UpdateSceneEntities":
        payload = data["payload"]
        entities = [
            SceneEntity(
                id=e["id"],
                position=e["position"],
                orientation=e["orientation"],
                entity_type=e["entity_type"],
                metadata=e.get("metadata", {}),
            )
            for e in payload["entities"]
        ]
        return cls(entities=entities)


@dataclass
class AttentionResultsMessage:
    """Python → Engine: attention evaluation results.

    Attributes:
        results: List of :class:`~vrspin.models.AttentionResult` objects.
    """

    results: List[AttentionResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "attention_results",
            "payload": {
                "results": [
                    {
                        "entity_id": r.entity_id,
                        "in_attention": r.in_attention,
                        "angular_distance": r.angular_distance,
                        "weight": r.weight,
                    }
                    for r in self.results
                ],
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttentionResultsMessage":
        payload = data["payload"]
        results = [
            AttentionResult(
                entity_id=r["entity_id"],
                in_attention=r["in_attention"],
                angular_distance=r["angular_distance"],
                weight=r.get("weight", 0.0),
            )
            for r in payload["results"]
        ]
        return cls(results=results)
