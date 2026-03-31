
┌──────────────────────────────────────────────────────┐
│                   VR Engine (Unity/Unreal)            │
│  ┌─────────┐  ┌──────────┐  ┌───────────────────┐   │
│  │ Headset │  │ Renderer │  │ Spatial Audio Eng │   │
│  └────┬────┘  └─────┬────┘  └────────┬──────────┘   │
│       │              │               │               │
│       │    ┌─────────┴───────────────┘               │
│       │    │  Bridge Layer (JSON/WebSocket/C API)     │
└───────┼────┼─────────────────────────────────────────┘
        │    │
        ▼    ▼
┌──────────────────────────────────────────────────────┐
│              VRSpin VR Module (Python)               │
│                                                      │
│  ┌────────────────┐  ┌──────────────────────┐        │
│  │ AttentionCone  │  │ SceneEntity (Node)   │        │
│  │  - origin_quat │  │  - position (vec3)   │        │
│  │  - half_angle  │  │  - orientation (quat) │       │
│  │  - contains()  │  │  - entity_type        │       │
│  │  - query()     │  │  - metadata           │       │
│  └───────┬────────┘  └──────────┬───────────┘        │
│          │                      │                    │
│  ┌───────┴──────────────────────┴───────────┐        │
│  │          AttentionManager                 │        │
│  │  - update(user_quat) → AttentionResult   │        │
│  │  - register_entity(SceneEntity)          │        │
│  │  - get_attended_entities()               │        │
│  └──────────────────────────────────────────┘        │
│                                                      │
│  ┌─────────────────┐  ┌────────────────────┐         │
│  │ NPCAttention    │  │ MultiHeadAttention │         │
│  │  Agent           │  │  - visual_cone     │         │
│  │  - perception   │  │  - audio_cone      │         │
│  │    _cone        │  │  - haptic_cone     │         │
│  │  - face_toward()│  │  - update()        │         │
│  │  - is_aware_of()│  │  - merge_results() │         │
│  └─────────────────┘  └────────────────────┘         │
│                                                      │
│  ┌──────────────────────────────────────────┐        │
│  │     Core SpinStep (existing, unchanged)   │        │
│  │  Node, quaternion_distance,               │        │
│  │  is_within_angle_threshold,               │        │
│  │  batch_quaternion_angle,                  │        │
│  │  DiscreteOrientationSet, rotate_quaternion │       │
│  └──────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────┘

Design Principles

    SpinStep core MUST be imported into module
    VRSpin module imports existing primitives
    Engine-agnostic bridge — JSON protocol for Unity/Unreal communication
    Stateless computation — each update() call is a pure function of current state