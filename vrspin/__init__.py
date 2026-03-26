"""VRSpin — SpinStep-powered "Look & Interact" VR demo.

Public API::

    from vrspin import AttentionCone, VRUser, NPC, NPCState
    from vrspin import InteractiveObject, AudioSource, KnowledgePanel, PanelPage
    from vrspin import VirtualPlaza, PlazaEvent
"""

from .cone import AttentionCone
from .user import VRUser
from .npc import NPC, NPCState
from .entities import InteractiveObject, AudioSource, KnowledgePanel, PanelPage
from .plaza import VirtualPlaza, PlazaEvent

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
]

__version__ = "0.1.0"
