"""VRSpin — SpinStep-powered "Look & Interact" VR demo.

Public API::

    from vrspin import AttentionCone, VRUser, NPC, NPCState
    from vrspin import InteractiveObject, AudioSource, KnowledgePanel, PanelPage
    from vrspin import VirtualPlaza, PlazaEvent
    from vrspin import SceneEntity, AttentionResult
    from vrspin import AttentionManager, NPCAttentionAgent, MultiHeadAttention
    from vrspin import (
        quaternion_distance, forward_vector_from_quaternion,
        direction_to_quaternion, slerp,
    )
    from vrspin import UpdateUserPose, UpdateSceneEntities, AttentionResultsMessage
"""

from .cone import AttentionCone
from .user import VRUser
from .npc import NPC, NPCState
from .entities import InteractiveObject, AudioSource, KnowledgePanel, PanelPage
from .plaza import VirtualPlaza, PlazaEvent
from .models import SceneEntity, AttentionResult
from .attention_manager import AttentionManager
from .npc_agent import NPCAttentionAgent
from .multi_head import MultiHeadAttention
from .math_utils import (
    quaternion_distance,
    forward_vector_from_quaternion,
    direction_to_quaternion,
    slerp,
)
from .bridge import UpdateUserPose, UpdateSceneEntities, AttentionResultsMessage

__all__ = [
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
    "SceneEntity",
    "AttentionResult",
    "AttentionManager",
    "NPCAttentionAgent",
    "MultiHeadAttention",
    "quaternion_distance",
    "forward_vector_from_quaternion",
    "direction_to_quaternion",
    "slerp",
    "UpdateUserPose",
    "UpdateSceneEntities",
    "AttentionResultsMessage",
]

__version__ = "0.1.0"
