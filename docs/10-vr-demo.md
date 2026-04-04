# VRSpin "Look & Interact" — VR Demo Guide

This document describes the "Look & Interact" demo scene, explains how to run it,
and walks through each interaction step.

---

## Concept

VRSpin simulates an **orientation-first** VR experience.
Everything in the virtual plaza reacts to a single input: **where the user is looking**
(their head quaternion).  No buttons, menus, or controllers needed — just natural
head-turning.

---

## Scene Layout

```
Plaza (root SpinStep Node)
├── NORTH   [  0°]  — Fountain + NPC Elena + Ambient Water Audio
├── NW      [ 70°]  — "VR Art" Knowledge Panel
├── WEST    [ 85°]  — "Digital Sculpture" Knowledge Panel
└── EAST    [-70°]  — Market Stand + NPC Kai + Market Music Audio
```

Orientations are chosen so that the panels and market stand are **outside** the user's
forward visual cone (60° half-angle) when facing north.  They only appear when the user
deliberately turns in their direction — exactly the intended "look to reveal" mechanic.

---

## Running the Demo

```bash
# Install dependencies
pip install -r requirements.txt

# Run the step-by-step interaction scenario
python demo_look_and_interact.py
```

Expected output:

```
────────────────────────────────────────────────────────────
 SpinStep VR Demo  —  'Look & Interact'
────────────────────────────────────────────────────────────

[Step 01] Maya enters the plaza facing NORTH (toward the Fountain)
  [VISUAL  ] (Fountain) 'Fountain' enters visual cone — highlighted ✦
  [AUDIO   ] (FountainAmbience) begins — soft water sounds (volume 0.70)
  ...
```

---

## Interaction Steps

### Step 1 — Facing North (0°)

The user faces the fountain.

- **Visual cone (60°)** highlights the Fountain.
- **Audio cone (120°)** starts FountainAmbience and MarketMusic.
- **Haptic cone (30°)** fires a controller pulse on the Fountain.
- **NPC Elena** notices the user and begins rotating toward them.
  After several ticks: `"Welcome to the plaza!"`

### Step 2 — Turn Left ~15°

The VR Art Knowledge Panel enters the user's visual cone.

- Panel appears: *"Virtual Reality Art: A New Medium"*

### Step 3 — Turn Left to ~70°

The Fountain leaves the visual cone.  The art panel shows page 2.
The Digital Sculpture panel becomes visible.
NPC Kai greets from the east market.

### Step 4 — Return to North

Fountain re-enters the visual cone.

### Step 5 — Turn Right to ~−70° (East)

The Market Stand enters the visual cone.  NPC Kai engages.
MarketMusic volume increases.

### Step 6 — Face North Again

Fountain re-highlighted.  Haptic pulse.

---

## Attention-Cone Mechanics

### Visual Cone (half-angle 60°)

Controls object highlighting and knowledge panel visibility.

```python
if user.visual_cone.is_in_cone(entity.orientation):
    entity.activate()      # highlight ✦
```

### Audio Cone (half-angle 120°)

Controls spatial audio volume.  Uses `cosine` falloff for natural roll-off.

```python
gain = user.audio_cone.attenuation(source.orientation)
source.set_volume(gain * source.base_volume)
```

### Haptic Cone (half-angle 30°)

Fires a controller pulse when an entity is nearly dead-ahead.

```python
if user.haptic_cone.is_in_cone(entity.orientation):
    events.append(PlazaEvent(tick, "haptic", entity.name, "Controller pulse"))
```

### NPC Perception (half-angle 120°)

Each NPC has its own `AttentionCone`.  When the user enters the NPC's cone,
the NPC smoothly SLERP-rotates toward the user over several ticks.

```python
if npc.user_in_cone(user):
    npc.tick(user)  # state: IDLE → NOTICING → ENGAGED → SPEAKING
```

---

## Scene-Tree Traversal

The plaza uses SpinStep's `QuaternionDepthIterator` to traverse the `Node` tree
using the user's orientation as the rotation step.  This naturally visits entities
that are spatially aligned with the current gaze direction.

```python
root = Node("plaza", user.orientation)
for node in QuaternionDepthIterator(root, user.orientation, max_depth=3):
    process(node)
```

---

## Multi-Head Attention

`VRUser` carries three independent `AttentionCone` objects simultaneously:

| Cone | Half-angle | SpinStep primitive |
|---|---|---|
| `visual_cone` | 60° | `is_within_angle_threshold` |
| `audio_cone` | 120° | `is_within_angle_threshold` |
| `haptic_cone` | 30° | `is_within_angle_threshold` |

All three are updated atomically when `user.set_orientation(q)` is called.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

138 tests across:
- `tests/test_vrspin.py` — AttentionCone, VRUser, entities, NPC, VirtualPlaza
- `tests/test_scene.py` — SceneEntity, AttentionManager, AttentionResult
- `tests/test_npc_agent.py` — NPCAttentionAgent, slerp, utility re-exports
- `tests/test_multihead.py` — MultiHeadAttention, merge_results
- `tests/test_plaza_visualization.py` — visualization logic, rendering

---

## See Also

- [API Reference](09-api-reference.md)
- [Capability Report](capability-report.md)
- [WebSocket Bridge README](../examples/vr_bridge_server_README.md)
