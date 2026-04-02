# Plaza Visualization — Desktop Demo

> **See VRSpin's SpinStep-powered attention cones in action — no VR headset required.**

The plaza visualization renders a top-down view of the VRSpin virtual plaza,
showing how SpinStep's quaternion-driven orientation framework determines which
entities are perceived by the user at any given head orientation.

---

## Quick Start

```bash
# Install dependencies
pip install -e .    # installs vrspin + spinstep
pip install matplotlib

# Generate static visualization frames
python examples/plaza_visualization.py --static

# Interactive mode (requires display)
python examples/plaza_visualization.py
```

---

## What It Shows

The visualization displays a bird's-eye view of the VRSpin plaza with:

| Element | Description |
|---|---|
| **User position** | Green circle at center with forward-direction arrow |
| **Visual cone** (green, 60°) | Narrow gaze focus — objects in this cone get highlighted |
| **Audio cone** (yellow, 120°) | Wide peripheral hearing — audio sources in range play |
| **Haptic cone** (purple, 30°) | Precise targeting — controller pulse for nearby objects |
| **Entities** | Objects, NPCs, audio sources, and knowledge panels with live status |
| **Info panel** | Real-time readout of all entity states, strengths, and events |
| **SpinStep panel** | Live readout of SpinStep primitive results (distances, traversals, spins) |

### Entity Types

- **◆ Interactive Objects** — Fountain, MarketStand — highlight when in visual cone
- **♀♂ NPCs** — Elena, Kai — color-coded by state (idle/noticing/engaged/speaking)
- **♫ Audio Sources** — FountainAmbience, MarketMusic — volume shown when playing
- **▣ Knowledge Panels** — VR Art, Digital Sculpture — appear when in visual cone

---

## SpinStep Primitives Used

The visualization directly imports and uses these SpinStep primitives:

| SpinStep Primitive | Usage in Visualization |
|---|---|
| `quaternion_from_euler(angles)` | Convert user yaw angle to head quaternion |
| `forward_vector_from_quaternion(q)` | Extract 3D gaze direction for forward arrow |
| `quaternion_distance(q1, q2)` | Compute angular distance to each entity |
| `is_within_angle_threshold(q, t, θ)` | Core cone membership test (via AttentionCone) |
| `batch_quaternion_angle(qs1, qs2, xp)` | Vectorised batch distance computation (via AttentionCone) |
| `get_relative_spin(nf, nt)` | Compute NPC-to-user rotation delta |
| `Node(name, orientation)` | Scene-tree nodes for entities and user |
| `QuaternionDepthIterator` | Traverse scene tree aligned with user gaze |

All results are displayed in the **SPINSTEP PRIMITIVES** section of the info panel.

---

## Interactive Controls

| Key | Action |
|---|---|
| `←` / `→` | Rotate user orientation by 10° |
| `1` | Toggle visual cone visibility |
| `2` | Toggle audio cone visibility |
| `3` | Toggle haptic cone visibility |
| `R` | Reset orientation to north (0°) |
| `Q` / `Esc` | Quit |

---

## Static Frame Generation

For CI pipelines, documentation, or headless environments:

```bash
# Default scenic tour (8 frames at key angles)
python examples/plaza_visualization.py --static

# Custom angles
python examples/plaza_visualization.py --static --angles "0,30,60,90,-45,-90"

# Custom output directory
python examples/plaza_visualization.py --static --output-dir ./docs/images
```

---

## Programmatic Usage

```python
from examples.plaza_visualization import compute_plaza_state, render_frame

# Compute plaza state at a specific user yaw
state = compute_plaza_state(user_yaw_deg=45.0)

# VRSpin entity states
print(state.active_objects)    # ['Fountain'] or []
print(state.visible_panels)    # ['VR Art'] when facing NW
print(state.npc_states)        # {'Elena': 'noticing', 'Kai': 'idle'}
print(state.audio_playing)     # {'FountainAmbience': 0.7, ...}
print(state.visual_strengths)  # {'Fountain': 0.85, ...}

# SpinStep primitive results
print(state.user_forward_vector)      # (0.0, 0.0, 1.0) gaze direction
print(state.entity_distances_deg)     # {'Fountain': 0.0, 'Kai': 70.0, ...}
print(state.npc_relative_spins_deg)   # {'Elena': 0.0, 'Kai': 70.0}
print(state.tree_attended_names)      # ['plaza', 'north_zone', ...]

# Render to file
render_frame(state, filepath="plaza_snapshot.png")
```

---

## How It Connects to SpinStep & VRSpin

The visualization showcases the full SpinStep → VRSpin pipeline:

1. **SpinStep `quaternion_from_euler`** converts user yaw to a quaternion
2. **SpinStep `forward_vector_from_quaternion`** extracts the 3D gaze direction
3. **VRSpin `VRUser.set_orientation`** updates all three attention cones
4. **SpinStep `QuaternionDepthIterator`** traverses the `Node` scene tree
5. **SpinStep `is_within_angle_threshold`** tests cone membership (via `AttentionCone`)
6. **SpinStep `batch_quaternion_angle`** computes vectorised distances (via `AttentionCone`)
7. **SpinStep `quaternion_distance`** measures per-entity angular distances
8. **SpinStep `get_relative_spin`** computes NPC rotation deltas

This is the same flow that the WebSocket bridge server
(`examples/vr_bridge_server.py`) follows for Unity/Unreal integration.

---

## Architecture

```
User Input (keyboard)
    │
    ▼
compute_plaza_state(yaw_deg)
    │
    ├── SpinStep: quaternion_from_euler() → user quaternion
    ├── SpinStep: forward_vector_from_quaternion() → gaze direction
    ├── VRSpin:   VirtualPlaza() ← SpinStep Node tree
    ├── VRSpin:   VRUser(orientation) ← 3 AttentionCones
    ├── VRSpin:   plaza.tick(user) ← full attention query
    ├── SpinStep: QuaternionDepthIterator → tree traversal
    ├── SpinStep: quaternion_distance() → per-entity distances
    └── SpinStep: get_relative_spin() → NPC rotation deltas
    │
    ▼
VisualizationState
    │
    ├── VRSpin:   active_objects, visible_panels, npc_states
    ├── VRSpin:   audio_playing, visual/audio/haptic strengths
    ├── SpinStep: tree_attended_names, entity_distances_deg
    └── SpinStep: npc_relative_spins_deg, user_forward_vector
    │
    ▼
render_frame(state)
    │
    ├── Top-down plaza map with cone wedges
    ├── Entity markers with state colours
    ├── Info panel with live readouts
    └── SpinStep primitives panel
```

---

## Requirements

- Python 3.9+
- matplotlib ≥ 3.5
- numpy, scipy (already required by vrspin)
- spinstep (installed automatically with vrspin)
- No VR headset, GPU, or special hardware needed
