## SpinStep Capability Mapping
### Existing Primitives → VR Concepts

SpinStep library    VRSpin Demo Usage
Node(name, orientation) 	Every scene entity (NPC, object, panel) is a Node with an orientation
quaternion_distance(q1, q2) 	Compute angular distance between user gaze and entity direction
is_within_angle_threshold(q_current, q_target, threshold) 	Core "is this entity inside my attention cone?" check
DiscreteOrientationSet.query_within_angle(quat, angle) 	Batch query: "which of N entities are inside my cone?"
rotate_quaternion(q, step) 	Smoothly update user/NPC orientation each frame
quaternion_from_euler(angles) 	Convert VR headset Euler angles to quaternion
batch_quaternion_angle(qs1, qs2, xp) 	Compute pairwise distances between user cone and all entities in one call
DiscreteOrientationSet.from_sphere_grid(n) 	Pre-compute NPC perception directions
QuaternionDepthIterator 	Traverse scene graph to find reachable entities in orientation space
get_relative_spin(nf, nt) 	Compute rotation needed for NPC to face user

### 2.2 Specific Mappings to Demo Mechanics

User Gaze → Attention Cone

Existing: is_within_angle_threshold(user_quat, entity_quat, cone_half_angle)
Usage:    For each entity in scene, check if it falls within user's visual cone

The user's VR headset provides a quaternion. SpinStep's is_within_angle_threshold() already performs the exact operation needed: "is entity B within angle θ of orientation A?"
NPC Attention → Reverse Cone Check

Existing: get_relative_spin(npc_node, user_node) → relative rotation quaternion
          quaternion_distance(npc_quat, user_quat) → angular distance
Usage:    NPC checks if user is within its own perception cone
          If yes → NPC rotates toward user using get_relative_spin()

Object Highlighting → Batch Query

Existing: batch_quaternion_angle(user_forward, all_entity_orientations, np)
Usage:    Single vectorized call returns (1×N) distance matrix
          Objects below threshold get highlighted

Knowledge Panels → Discrete Orientation Set

Existing: DiscreteOrientationSet.from_custom(panel_directions)
          .query_within_angle(user_quat, reading_cone_angle)
Usage:    Panels placed at known orientations; user rotates into their cone to read