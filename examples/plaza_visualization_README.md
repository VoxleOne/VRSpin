# Plaza Visualization — Desktop Demo

> **See VRSpin's attention cones in action — no VR headset required.**

The plaza visualization renders a top-down view of the VRSpin virtual plaza,
showing how orientation-driven attention cones determine which entities are
perceived by the user at any given head orientation.

---

## Quick Start

```bash
# Install dependencies
pip install -e .
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

### Entity Types

- **◆ Interactive Objects** — Fountain, MarketStand — highlight when in visual cone
- **♀♂ NPCs** — Elena, Kai — color-coded by state (idle/noticing/engaged/speaking)
- **♫ Audio Sources** — FountainAmbience, MarketMusic — volume shown when playing
- **▣ Knowledge Panels** — VR Art, Digital Sculpture — appear when in visual cone

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

print(state.active_objects)    # ['Fountain'] or []
print(state.visible_panels)    # ['VR Art'] when facing NW
print(state.npc_states)        # {'Elena': 'noticing', 'Kai': 'idle'}
print(state.audio_playing)     # {'FountainAmbience': 0.7, ...}
print(state.visual_strengths)  # {'Fountain': 0.85, ...}

# Render to file
render_frame(state, filepath="plaza_snapshot.png")
```

---

## How It Connects to VRSpin

The visualization uses the exact same VRSpin API that a real VR engine would:

1. Creates a `VirtualPlaza` and `VRUser`
2. Sets the user's head orientation via `user.set_orientation(quaternion)`
3. Calls `plaza.tick(user)` to process one simulation frame
4. Reads entity states (`obj.active`, `panel.visible`, `npc.state`, `audio.volume`)
5. Uses `AttentionCone.attenuation()` to compute per-entity attention strengths

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
    ├── VirtualPlaza()     ← SpinStep Node tree
    ├── VRUser(orientation) ← 3 AttentionCones
    ├── plaza.tick(user)   ← full attention query
    │
    ▼
VisualizationState
    │
    ├── active_objects, visible_panels
    ├── npc_states, audio_playing
    ├── visual/audio/haptic strengths
    │
    ▼
render_frame(state)
    │
    ├── Top-down plaza map with cone wedges
    ├── Entity markers with state colours
    └── Info panel with live readouts
```

---

## Requirements

- Python 3.9+
- matplotlib ≥ 3.5
- numpy, scipy (already required by vrspin)
- No VR headset, GPU, or special hardware needed
