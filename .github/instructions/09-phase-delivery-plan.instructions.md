## Phased Delivery Plan

### Phase 1: Core Attention Module

Goal: AttentionCone and utility functions 

Task 	File 	LOC Est. 	Dependencies
forward_vector_from_quaternion() 	spinstep/utils/quaternion_utils.py 	~10 	scipy
slerp() 	spinstep/utils/quaternion_utils.py 	~15 	scipy
direction_to_quaternion() 	spinstep/utils/quaternion_utils.py 	~10 	scipy
AttentionCone class 	vrspinstep/attention.py 	~80 	numpy, spinstep

Tests for above

Deliverable: AttentionCone usable standalone — no VR engine needed.

### Phase 2: Scene & NPC Layer

Goal: SceneEntity, AttentionManager, NPCAttentionAgent.

Task 	File 	LOC Est. 	Dependencies
SceneEntity class 	vrspin/scene.py 	~60 	spinstep
AttentionResult dataclass 	vrspin/scene.py 	~20 	—
AttentionManager class 	vrspin/scene.py 	~80 	AttentionCone
NPCAttentionAgent class 	vrspin/npc.py 	~90 	slerp, AttentionCone
Tests for scene module 	tests/test_scene.py 	~100 	pytest
Tests for NPC module 	tests/test_npc.py 	~80 	pytest

Deliverable: Full scene graph with NPC attention — testable in pure Python.

### Phase 3: Multi-Head & Audio

Goal: MultiHeadAttention and audio gain computation.

Task 	File 	LOC Est. 	Dependencies
MultiHeadAttention class 	vrspin/multihead.py 	~70 	AttentionCone
Audio gain helper 	vrspin/attention.py 	~20 	AttentionCone
Tests for multi-head 	tests/test_multihead.py 	~80 	pytest

Deliverable: Multi-modal attention queries — visual, audio, haptic.

### Phase 4: VR Engine Bridge

Goal: WebSocket server for Unity/Unreal integration.

Task 	File 	LOC Est. 	Dependencies
WebSocket server 	examples/vr_bridge_server.py 	~120 	asyncio, websockets
JSON protocol handlers 	examples/vr_bridge_server.py 	~60 	json
Unity client script 	examples/unity/SpinStepClient.cs 	~80 	Unity WebSocket
Demo scene description 	examples/vr_plaza_demo.py 	~100 	vrspin

Deliverable: Working prototype — Unity sends head quaternion, VRSpin responds with attention results.

### Phase 5: Polish & Documentation

Task 	File
VR demo documentation 	docs/10-vr-demo.md
API reference updates 	docs/09-api-reference.md
Demo video script / README 	examples/vr_bridge_server_README.md
Performance benchmarks 	benchmark/vr_attention_benchmark.py