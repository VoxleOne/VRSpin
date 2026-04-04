"""VRSpin — SpinStep-powered "Look & Interact" VR demo.

Public API::

    from vrspin import AttentionCone, VRUser, NPC, NPCState
    from vrspin import InteractiveObject, AudioSource, KnowledgePanel, PanelPage
    from vrspin import VirtualPlaza, PlazaEvent
    from vrspin import SceneEntity, AttentionManager, AttentionResult
    from vrspin import NPCAttentionAgent
    from vrspin import forward_vector_from_quaternion, direction_to_quaternion
    from vrspin import angle_between_directions, slerp
"""

from .cone import AttentionCone
from .user import VRUser
from .npc import NPC, NPCState, NPCAttentionAgent
from .entities import InteractiveObject, AudioSource, KnowledgePanel, PanelPage
from .plaza import VirtualPlaza, PlazaEvent
from .scene import SceneEntity, AttentionManager, AttentionResult

# Re-export utility functions from SpinStep for convenience.
from spinstep.utils import (
    forward_vector_from_quaternion,
    direction_to_quaternion,
    angle_between_directions,
)

# slerp is not provided by SpinStep — implemented locally.
from .utils import slerp

__all__ = [
    # Existing public API
    "AttentionCone",
    "VRUser",
    "NPC",
    "NPCState",
    "InteractiveObject",
    "AudioSource",
    "KnowledgePanel",
    "PanelPage",
    "VirtualPlaza",
    "PlazaEvent",
    # Instruction-style additions
    "SceneEntity",
    "AttentionManager",
    "AttentionResult",
    "NPCAttentionAgent",
    # Utility functions
    "forward_vector_from_quaternion",
    "direction_to_quaternion",
    "angle_between_directions",
    "slerp",
]

__version__ = "0.1.0"
