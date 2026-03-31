## Public API Design
### Minimal Public Interface

The VRSpin demo module should expose exactly these classes and functions:

#### vrspin/__init__.py:

__all__ = [
    # New VRSpin/attention module
    "AttentionCone",
    "SceneEntity",
    "AttentionManager",
    "NPCAttentionAgent",

    # New utility functions
    "forward_vector_from_quaternion",
    "slerp",
]

### Import Patterns

#### Basic attention check
from vrspin import AttentionCone
cone = AttentionCone(user_quat, half_angle=0.5)

#### Full scene management
from vrspin import SceneEntity, AttentionManager
from vrspin.npc import NPCAttentionAgent

#### Multi-head (advanced, optional import)
from vrspin.multihead import MultiHeadAttention

#### Utility functions
from spinstep.utils import forward_vector_from_quaternion, slerp

### 6.3 Design Constraints

    All new classes compose existing SpinStep primitives (no reimplementation)
    All orientations use the existing [x, y, z, w] quaternion convention
    All angles in radians (matching existing API)
    No side effects at import time (matching existing convention)
    Type hints on all public methods (matching existing convention)
    Google-style docstrings (matching existing convention)