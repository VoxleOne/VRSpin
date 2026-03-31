## Utility Analysis

### Utilities in spinstep/quaternion_utils.py

    "quaternion_distance",
    "rotate_quaternion",
    "is_within_angle_threshold",
    "quaternion_conjugate",
    "quaternion_multiply",
    "rotation_matrix_to_quaternion",
    "get_relative_spin",
    "get_unique_relative_spins",
    "forward_vector_from_quaternion",
    "direction_to_quaternion",
    "angle_between_directions",

### Utilities  additions to VRSpin/ from spinstep

Function 	Purpose 	Complexity
forward_vector_from_quaternion(q) 	Extract the forward (look) direction from a quaternion 	Trivial — R.from_quat(q).apply([0, 0, -1])
direction_to_quaternion(direction) 	Convert a 3D direction vector to an orientation quaternion 	Small — R.align_vectors()
angle_between_directions(d1, d2) 	Angular distance between two direction vectors 	Trivial — arccos(dot)
slerp(q1, q2, t) 	Spherical linear interpolation for smooth NPC turning 	Small — scipy.spatial.transform.Slerp

### New Module: Attention Cone (VRSpin/attention.py)

SpinStep's traversal iterators implicitly define orientation cones via angle thresholds. The VRSpin demo requires making this concept explicit and reusable as a first-class object.

Key insight: AttentionCone is essentially a wrapper around is_within_angle_threshold() and batch_quaternion_angle() with a fixed cone geometry.

### New Module: Scene Integration (VRSpin/scene.py)

The existing spinstep Node class stores (name, orientation, children). The VRSpin demo needs:

    3D position (for spatial audio distance attenuation)
    Entity type (NPC, object, panel)
    Activation state (highlighted, active, idle)
    Custom metadata

This should be a SpinStep Node

### New Module: NPC Attention Agent (VRSpin/npc.py)

NPC behavior is fundamentally:

    Check if any user is inside my perception cone
    If yes → compute relative rotation to face them (get_relative_spin)
    Smoothly interpolate my orientation toward the user (slerp)

All of this imports existing SpinStep primitives.

### New Module: Multi-Head Attention (optional, VRSpin/multihead.py)

For the advanced twist — multiple independent cones (visual, audio, haptic) with independent thresholds and different update rates."quaternion_from_euler",

