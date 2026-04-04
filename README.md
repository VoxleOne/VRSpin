# VRSpin — "Look & Interact" VR Demo

> **Attention drives perception and interaction.**  
> Objects, audio, NPCs, and virtual knowledge appear based on **orientation cones** — powered by [SpinStep](https://github.com/VoxleOne/SpinStep).

---

## Concept

VRSpin is a Python simulation of an **orientation-first VR experience**.

The entire scene reacts to a single input: **where you're looking** (head quaternion). No menus, no clicks.

The demo models a small virtual plaza (Horizon-Worlds style) populated with NPCs,
interactive objects, spatial audio sources, and floating knowledge panels.  
Every perceivable thing in the scene is guarded by a **SpinStep attention cone**.

---

## Scene Layout

```
Plaza (root Node)
├── NORTH  [0°]  — Fountain + NPC Elena + Ambient Water
├── NW    [70°]  — "VR Art" Knowledge Panel
├── WEST  [85°]  — "Digital Sculpture" Knowledge Panel
└── EAST [-70°]  — Market Stand + NPC Kai + Market Music
```

---

## How SpinStep Powers the Demo

> **Orientation space, not coordinate space.**  
> Unlike conventional scene-graph systems — which place entities in ℝ³ coordinate
> space using position vectors and transform matrices — SpinStep represents every
> node as a **unit quaternion on S³** (the 3-sphere of rotations). A SpinStep
> `Node` carries only a name, an orientation quaternion `[x, y, z, w]`, and its
> children — no position, no scale, no transform hierarchy. Traversal, proximity,
> and visibility are all determined by **angular distance between orientations**,
> not Euclidean distance between points. This is a core design feature of SpinStep
> and is what makes orientation-driven queries (attention cones, gaze matching,
> depth-first rotation stepping) natural first-class operations rather than
> after-the-fact projections onto a spatial scene.

| Mechanic | SpinStep primitive |
|---|---|
| User gaze | `VRUser` head quaternion → `AttentionCone.get_forward_vector()` |
| Scene-tree traversal | `QuaternionDepthIterator` walks `Node` tree aligned with gaze |
| Batch cone queries | `DiscreteOrientationSet.query_within_angle()` |
| Object highlight / panel reveal | `AttentionCone.is_in_cone()` per entity orientation |
| NPC rotation toward user | quaternion SLERP via `scipy.spatial.transform.Slerp` |
| Multi-modal perception | three independent `AttentionCone` instances per user |

### Multi-Head SpinStep

Each `VRUser` carries **three attention cones** — one per sensory modality:

| Cone | Half-angle | Purpose |
|---|---|---|
| `visual_cone` | 60° | Object highlights, knowledge panels |
| `audio_cone` | 120° | Spatial audio boost / attenuation |
| `haptic_cone` | 30° | Controller pulse (precise targeting) |

---

## Demo Output

```
────────────────────────────────────────────────────────────
 SpinStep VR Demo  —  'Look & Interact'
────────────────────────────────────────────────────────────

[Step 01] Maya enters the plaza facing NORTH (toward the Fountain)
  [VISUAL  ] (Fountain) 'Fountain' enters visual cone — highlighted ✦
  [AUDIO   ] (FountainAmbience) 'FountainAmbience' begins — soft water sounds (volume 0.70)
  [AUDIO   ] (MarketMusic) 'MarketMusic' begins — lively merchant tune (volume 0.90)
  [NPC     ] (Elena) NPC 'Elena' notices 'Maya' — begins rotating
  [NPC     ] (Elena) NPC 'Elena' says: "Welcome to the plaza! The fountain has been here for ages."
  [HAPTIC  ] (Fountain) Controller pulse — 'Fountain' is directly ahead

[Step 02] Maya pivots LEFT ~15° — VR Art knowledge panel becomes visible
  [KNOWLEDGE] (VR Art) Panel 'VR Art' appears — Virtual Reality Art: …

[Step 03] Maya rotates further LEFT to 70° — panel content updates
  [VISUAL  ] (Fountain) 'Fountain' leaves visual cone — highlight off
  [KNOWLEDGE] (VR Art) Panel 'VR Art' → page 2: Interaction Design in VR
  [KNOWLEDGE] (Digital Sculpture) Panel 'Digital Sculpture' appears — …
  [NPC     ] (Kai) NPC 'Kai' says: "Step right up! Best wares in the metaverse!"

[Step 05] Maya turns RIGHT toward the East Market Stand (−70°)
  [VISUAL  ] (MarketStand) 'MarketStand' enters visual cone — highlighted ✦
  [HAPTIC  ] (MarketStand) Controller pulse — 'MarketStand' is directly ahead

[Step 06] Maya faces NORTH directly at the Fountain (haptic pulse)
  [VISUAL  ] (Fountain) 'Fountain' enters visual cone — highlighted ✦
  [HAPTIC  ] (Fountain) Controller pulse — 'Fountain' is directly ahead
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo
python demo_look_and_interact.py
```

### Desktop Visualization (no VR headset needed)

```bash
pip install matplotlib

# Interactive mode (rotate with arrow keys)
python examples/plaza_visualization.py

# Generate static PNG frames
python examples/plaza_visualization.py --static
```

See [examples/plaza_visualization_README.md](examples/plaza_visualization_README.md) for full documentation.

### Run Tests

```bash
pip install pytest
python -m pytest tests/
```

---

## Package Structure

```
vrspin/
  __init__.py        — public API exports
  architecture.yaml  — architectural specification
  cone.py            — AttentionCone: quaternion orientation + half-angle membership test
  user.py            — VRUser: head orientation + 3 multi-modal attention cones
  npc.py             — NPC: SpinStep Node + perception cone + SLERP attention machine
  entities.py        — InteractiveObject, AudioSource, KnowledgePanel, PanelPage
  scene.py           — SceneEntity, AttentionManager, AttentionResult
  multihead.py       — MultiHeadAttention: multi-modal cone queries
  plaza.py           — VirtualPlaza: scene-tree + simulation tick engine
  utils.py           — slerp and helper functions
demo_look_and_interact.py  — runnable step-by-step interaction scenario
examples/
  plaza_visualization.py       — interactive top-down attention-cone visualization
  plaza_visualization_README.md — documentation for the visualization demo
  vr_bridge_server.py          — WebSocket bridge for Unity/Unreal integration
  vr_bridge_server_README.md   — documentation for the bridge server
  unity/SpinStepClient.cs      — Unity C# client sketch
tests/
  test_vrspin.py               — 59 unit tests covering core modules
  test_scene.py                — 16 tests for scene layer
  test_npc_agent.py            — 16 tests for NPC agent and utilities
  test_multihead.py            — 10 tests for multi-head attention
  test_plaza_visualization.py  — 27 tests for visualization logic
docs/
  09-api-reference.md          — complete API reference
  10-vr-demo.md                — VR demo guide
  capability-report.md         — architecture assessment
benchmark/
  vr_attention_benchmark.py    — performance benchmarks
```

---

## Technical Stack

| Layer | Technology |
|---|---|
| Orientation math | [SpinStep](https://github.com/VoxleOne/SpinStep) `Node`, `QuaternionDepthIterator`, `DiscreteOrientationSet` |
| Quaternion algebra | `scipy.spatial.transform.Rotation`, `Slerp` |
| Numerical arrays | `numpy` |
| Target runtime | Python 3.9+ |
| VR engine (production) | Unity3D or Unreal Engine (this repo = Python simulation layer) |

---

## Architecture

```
VRUser (head quaternion)
    │
    ├── visual_cone  ──→ InteractiveObject.activate() / KnowledgePanel.show()
    ├── audio_cone   ──→ AudioSource.start() / set_volume()
    └── haptic_cone  ──→ PlazaEvent(haptic, "Controller pulse")

VirtualPlaza.tick(user)
    │
    ├── QuaternionDepthIterator(root, user.orientation) — tree traversal
    ├── for each entity: AttentionCone.is_in_cone(entity.orientation)
    └── NPC.tick(user) → SLERP toward user, state machine: IDLE→NOTICING→ENGAGED→SPEAKING
```

---

## License

MIT — see [LICENSE](LICENSE)

