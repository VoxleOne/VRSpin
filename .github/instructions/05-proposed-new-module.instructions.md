## Proposed New Modules

### vrspin/attention.py — Attention Cone

"""Orientation-based attention cone for spatial perception."""

class AttentionCone:
    """A directional attention cone defined by an origin quaternion and half-angle.

    The cone represents a region of orientation space centered on
    a quaternion direction. Entities whose direction falls within
    the half-angle are considered "attended."
    
    Args:
        origin_quat: Center orientation of the cone [x, y, z, w].
        half_angle: Half-angle of the cone in radians.
        falloff: Optional distance-based attenuation ('linear', 'cosine', None).
    
    Example::
    
        from vrspin.attention import AttentionCone
        import numpy as np
    
        cone = AttentionCone([0, 0, 0, 1], half_angle=0.5)
        cone.update_origin([0, 0, 0.1, 0.995])
        entity_quat = [0.1, 0, 0, 0.995]
        print(cone.contains(entity_quat))   # True/False
        print(cone.attenuation(entity_quat))  # 0.0–1.0 strength
    """
    
    def __init__(self, origin_quat, half_angle, falloff=None): ...
    def update_origin(self, new_quat): ...
    def contains(self, target_quat) -> bool: ...
    def attenuation(self, target_quat) -> float: ...
    def query_batch(self, entity_quats) -> np.ndarray: ...
    def query_batch_with_attenuation(self, entity_quats) -> np.ndarray:

Implementation notes:

contains() delegates to is_within_angle_threshold(self.origin, target, self.half_angle)
    query_batch() delegates to batch_quaternion_angle(origin, entities, np) then filters
    attenuation() returns 1.0 - (distance / half_angle) clamped to [0, 1] for linear falloff

### vrspin/scene.py — Scene Entity

"""Scene entity with position, orientation, and metadata for VR integration."""

class SceneEntity(Node):
    """A scene-graph node with 3D position and VR-specific metadata.

    Extends Node with position, entity type, and activation state
    needed for VR scene management.
    
    Args:
        name: Entity identifier.
        orientation: Quaternion [x, y, z, w].
        position: 3D world position [x, y, z].
        entity_type: One of 'npc', 'object', 'panel', 'audio_source'.
        metadata: Arbitrary key-value data for the entity.
    """
    
    def __init__(self, name, orientation, position, entity_type, metadata=None): ...
    
    @property
    def direction_quaternion(self) -> np.ndarray: ...
    
    def distance_to(self, other) -> float: ...


class AttentionManager:
    """Manages attention queries across all scene entities.

    Maintains a registry of SceneEntity instances and efficiently
    queries which entities fall within an attention cone each frame.
    
    Args:
        entities: Initial list of scene entities.
    
    Example::
    
        manager = AttentionManager(entities)
        result = manager.update(user_head_quaternion, cone_half_angle=0.5)
        for entity, strength in result.attended:
            print(f"{entity.name}: {strength:.2f}")
    """
    
    def __init__(self, entities=None): ...
    def register_entity(self, entity): ...
    def unregister_entity(self, name): ...
    def update(self, user_quat, cone_half_angle) -> AttentionResult: ...

class AttentionResult:
    """Result of an attention query — attended entities with attenuation values."""

    attended: list  # List of (SceneEntity, float) tuples
    unattended: list  # Entities outside the cone

### vrspin/npc.py — NPC Attention Agent

"""NPC behavior driven by orientation-based attention."""

class NPCAttentionAgent:
    """An NPC that perceives and reacts to entities within its attention cone.

    The NPC has its own perception cone. When a user enters the cone,
    the NPC smoothly rotates toward them. When the user leaves, the
    NPC returns to its idle orientation.
    
    Args:
        entity: The SceneEntity representing this NPC.
        perception_half_angle: Half-angle of the NPC's perception cone (radians).
        turn_speed: Interpolation factor for smooth turning (0–1 per update).
        idle_orientation: Default orientation when no user is attended.
    
    Example::
    
        npc = NPCAttentionAgent(npc_entity, perception_half_angle=0.8)
        npc.update(user_positions_and_orientations, dt=0.016)
        new_orientation = npc.entity.orientation  # updated
    """
    
    def __init__(self, entity, perception_half_angle, turn_speed=0.1,
                 idle_orientation=None): ...
    def is_aware_of(self, target_quat) -> bool: ...
    def update(self, targets, dt) -> None: ...
    def face_toward(self, target_quat, t) -> None: ...

Implementation notes:

    is_aware_of() uses is_within_angle_threshold(self.orientation, target, self.half_angle)
    face_toward() uses scipy Slerp for smooth quaternion interpolation
    update() finds closest target in cone, then calls face_toward()

### vrspin/multihead.py — Multi-Head Attention (Advanced)

"""Multi-modal attention with independent cones per modality."""

class MultiHeadAttention:
    """Multiple independent attention cones for different modalities.

    Each 'head' is an AttentionCone with its own half-angle and
    falloff, representing a different sensory channel (visual, audio,
    haptic).
    
    Args:
        heads: Dict mapping modality names to AttentionCone instances.
    
    Example::
    
        multi = MultiHeadAttention({
            'visual': AttentionCone(user_quat, half_angle=0.5),
            'audio':  AttentionCone(user_quat, half_angle=1.2),
            'haptic': AttentionCone(user_quat, half_angle=0.3),
        })
        results = multi.update(user_quat, entities)
        # results['visual'] → entities in visual cone
        # results['audio']  → entities in wider audio cone
    """
    
    def __init__(self, heads): ...
    def update(self, origin_quat, entities) -> dict: ...
    def merge_results(self, strategy='union') -> list: ...
