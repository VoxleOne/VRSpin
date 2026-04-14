# API Reference — VRSpin

Complete reference for the `vrspin` public API.

---

## `vrspin.cone` — AttentionCone

```python
from vrspin import AttentionCone
```

### `AttentionCone`

```
AttentionCone(orientation, half_angle_rad=None, label='visual', *, half_angle=None, falloff=None)
```

A directional attention cone defined by a quaternion orientation and a half-aperture angle.
Wraps SpinStep's `is_within_angle_threshold` and `batch_quaternion_angle` into a reusable
first-class object.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `orientation` | array-like `[x, y, z, w]` | Quaternion for the cone's pointing direction |
| `half_angle_rad` | `float` | Half-aperture in radians (positional style) |
| `half_angle` | `float` | Alias for `half_angle_rad` (keyword style) |
| `label` | `str` | Modality label, e.g. `"visual"`, `"audio"`, `"haptic"` |
| `falloff` | `str \| None` | Attenuation curve: `'linear'`, `'cosine'`, or `None` (step) |

**Attributes**

| Name | Type | Description |
|---|---|---|
| `orientation` | `np.ndarray (4,)` | Normalised quaternion `[x, y, z, w]` |
| `half_angle` | `float` | Half-aperture in radians |
| `label` | `str` | Modality label |
| `falloff` | `str \| None` | Attenuation curve |

**Methods**

---

#### `update_orientation(orientation)`

Update the cone's pointing direction.

```python
cone.update_orientation([0, 0, 0.1, 0.995])
```

---

#### `update_origin(new_quat)`

Alias for `update_orientation`. Matches the instruction-style API.

---

#### `is_in_cone(target_quat) → bool`

Return `True` if `target_quat` falls within the cone's aperture.

```python
cone.is_in_cone([0, 0, 0.05, 0.999])  # True — nearly aligned
```

---

#### `contains(target_quat) → bool`

Alias for `is_in_cone`.

---

#### `attenuation(target_quat) → float`

Return a `[0, 1]` attention strength.

- `1.0` at the cone centre.
- `0.0` at or beyond the cone edge.
- Decay curve determined by `falloff`.

```python
strength = cone.attenuation([0, 0, 0.2, 0.98])  # e.g. 0.72
```

---

#### `query_batch(entity_quats) → np.ndarray`

Return a boolean mask `(N,)` indicating which quaternions are inside the cone.

```python
mask = cone.query_batch(np.array([[0,0,0,1], [0,0,0.9,0.44]]))
```

---

#### `query_batch_with_attenuation(entity_quats) → np.ndarray`

Return per-entity attenuation values `(N,)`.  Outside-cone entries are `0.0`.

---

#### `filter_within_cone(orientation_set) → np.ndarray`

Return indices of orientations from a `DiscreteOrientationSet` inside this cone.

---

#### `get_forward_vector() → np.ndarray`

Return the 3-D unit vector `(3,)` this cone is pointing toward.

---

#### `angular_distance_to(target_quat) → float`

Return the geodesic angular distance in radians `[0, π]` to `target_quat`.

---

## `vrspin.scene` — Scene Layer

```python
from vrspin import SceneEntity, AttentionManager, AttentionResult
```

### `SceneEntity`

```
SceneEntity(name, orientation, position=(0,0,0), entity_type='object', metadata=None)
```

A SpinStep `Node` subclass extended with 3-D position and VR metadata.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `name` | `str` | Entity identifier |
| `orientation` | array-like `[x,y,z,w]` | Initial orientation quaternion |
| `position` | array-like `[x,y,z]` | World-space position (default: origin) |
| `entity_type` | `str` | `'npc'`, `'object'`, `'panel'`, `'audio_source'` |
| `metadata` | `dict \| None` | Arbitrary key-value data |

**Properties / Methods**

| Name | Returns | Description |
|---|---|---|
| `direction_quaternion` | `np.ndarray (4,)` | Alias for `orientation` |
| `distance_to(other)` | `float` | Euclidean distance to another `SceneEntity` |

---

### `AttentionResult`

Dataclass holding the result of an attention query.

```python
@dataclass
class AttentionResult:
    attended: list[tuple[SceneEntity, float]]   # sorted by descending strength
    unattended: list[SceneEntity]
```

---

### `AttentionManager`

```
AttentionManager(entities=None)
```

Registry of `SceneEntity` objects queried against an attention cone each frame.

**Methods**

| Name | Description |
|---|---|
| `register_entity(entity)` | Add a `SceneEntity` |
| `unregister_entity(name)` | Remove by name |
| `update(user_quat, cone_half_angle, falloff='linear') → AttentionResult` | Query all entities |
| `get_attended_entities() → list[SceneEntity]` | Return entities from the most recent `update()` that were attended |

```python
manager = AttentionManager([fountain, vendor])
result = manager.update(user_quat, cone_half_angle=np.radians(45))
for entity, strength in result.attended:
    print(f"{entity.name}: {strength:.2f}")
```

---

### `Observer` Protocol

```python
from vrspin.scene import Observer
```

Any object that exposes `orientation` (a quaternion) and `attention_cones` (a dict of
`AttentionCone` instances) satisfies the `Observer` protocol and can be passed to
`AttentionManager.update_observers()`.

**Properties**

| Name | Type | Description |
|---|---|---|
| `orientation` | `np.ndarray (4,)` | Current orientation quaternion `[x, y, z, w]` |
| `attention_cones` | `dict[str, AttentionCone]` | Named perception cones (e.g. `'visual'`, `'audio'`) |

**Implemented by**

- `NPC` — via the `attention_cones` property (perception cone keyed as `'perception'`)
- `VRUser` — via `visual_cone`, `audio_cone`, `haptic_cone`

```python
# Any object satisfying the protocol works as an observer
class CustomObserver:
    def __init__(self, orientation, cones):
        self.orientation = orientation
        self.attention_cones = cones
```

---

#### `AttentionManager.update_observers(observers, cone_half_angle, falloff='linear') → dict[str, AttentionResult]`

Evaluate attention for multiple observers simultaneously. Each observer's cones are
queried against the registered entities.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `observers` | `list[Observer]` | Objects satisfying the `Observer` protocol |
| `cone_half_angle` | `float` | Default half-angle in radians (used when a cone doesn't specify one) |
| `falloff` | `str \| None` | Attenuation curve: `'linear'` (default), `'cosine'`, or `None` |

**Returns** `dict[str, AttentionResult]` — keyed by observer name.

```python
results = manager.update_observers([npc1, npc2, user], cone_half_angle=np.radians(60))
for observer_id, result in results.items():
    print(f"{observer_id}: {len(result.attended)} entities attended")
```

---

#### `NPC.observe(entities, cone_half_angle=None) → AttentionResult`

Run the NPC's perception cone against the given entities and return an `AttentionResult`.
This makes the NPC an active observer rather than a passive target.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `entities` | `list[SceneEntity]` | Scene entities to evaluate |
| `cone_half_angle` | `float \| None` | Override half-angle in radians; defaults to the NPC's perception cone |

**Returns** `AttentionResult` — attended and unattended entities from the NPC's point of view.

```python
result = npc.observe(entities)
for entity, strength in result.attended:
    print(f"  {entity.name}: {strength:.2f}")
```

---

#### `NPC.attention_cones` — property

```python
cones = npc.attention_cones  # dict[str, AttentionCone]
```

Exposes the NPC's perception cones as a dictionary, satisfying the `Observer` protocol.
Typically contains a single `'perception'` cone derived from `perception_half_angle`.

---

## `vrspin.npc` — NPC Agents

```python
from vrspin import NPC, NPCState, NPCAttentionAgent
```

### `NPCAttentionAgent`

```
NPCAttentionAgent(entity, perception_half_angle, turn_speed=0.1, idle_orientation=None)
```

Lightweight engine-agnostic NPC that perceives and reacts to targets in its cone.

**Methods**

| Name | Returns | Description |
|---|---|---|
| `is_aware_of(target_quat)` | `bool` | Perception-cone membership test |
| `face_toward(target_quat, t)` | `None` | SLERP entity orientation toward target by fraction `t` |
| `update(targets, dt)` | `None` | Per-frame update; rotates toward closest in-cone target or returns to idle |

---

### `NPC`

```
NPC(name, orientation, perception_half_angle=120°, slerp_speed=0.15, greeting='')
```

Full NPC with state machine and greeting system.

**States:** `IDLE → NOTICING → ENGAGED → SPEAKING`

**Methods**

| Name | Returns | Description |
|---|---|---|
| `user_in_cone(user)` | `bool` | Check if a `VRUser` is in perception cone |
| `tick(user)` | `list[str]` | Advance state machine; returns event strings |

---

### `NPCState`

Enum: `IDLE`, `NOTICING`, `ENGAGED`, `SPEAKING`.

---

## `vrspin.multihead` — Multi-Head Attention

```python
from vrspin.multihead import MultiHeadAttention
```

### `MultiHeadAttention`

```
MultiHeadAttention(heads: dict[str, AttentionCone])
```

Multiple independent attention cones, one per sensory modality.

**Methods**

| Name | Returns | Description |
|---|---|---|
| `update(origin_quat, entities)` | `dict[str, list[(SceneEntity, float)]]` | Query all heads; updates cone origins first |
| `merge_results(strategy='union')` | `list[SceneEntity]` | Merge last results; strategy `'union'` or `'intersection'` |

```python
multi = MultiHeadAttention({
    'visual': AttentionCone(user_quat, half_angle=np.radians(45)),
    'audio':  AttentionCone(user_quat, half_angle=np.radians(90)),
    'haptic': AttentionCone(user_quat, half_angle=np.radians(20)),
})
results = multi.update(user_quat, entities)
all_attended = multi.merge_results(strategy='union')
```

---

## `vrspin.user` — VR User

```python
from vrspin import VRUser
```

### `VRUser`

```
VRUser(name, orientation=(0,0,0,1))
```

A VR user with three pre-configured attention cones.

| Cone | Half-angle | Purpose |
|---|---|---|
| `visual_cone` | 60° | Object highlights, panels |
| `audio_cone` | 120° | Spatial audio |
| `haptic_cone` | 30° | Precise touch feedback |

**Methods**

| Name | Description |
|---|---|
| `set_orientation(orientation)` | Update head orientation and all three cones |
| `rotate_by(delta_quat)` | Apply a relative rotation step |
| `sees(entity)` | `True` if entity is in visual cone |
| `hears(entity)` | `True` if entity is in audio cone |
| `feels(entity)` | `True` if entity is in haptic cone |
| `get_forward_vector()` | 3-D forward direction |
| `cone_for(modality)` | Return cone by name string |

---

## `vrspin.entities` — Scene Entities

```python
from vrspin import InteractiveObject, AudioSource, KnowledgePanel, PanelPage
```

### `InteractiveObject`

SpinStep `Node` with activation state and highlight flag.

| Method | Description |
|---|---|
| `activate()` | Set `active=True`, `highlighted=True` |
| `deactivate()` | Set `active=False`, `highlighted=False` |

### `AudioSource`

SpinStep `Node` with volume and playback state.

| Method | Description |
|---|---|
| `start(volume=None)` | Begin playback at given or base volume |
| `stop()` | Stop playback |
| `set_volume(v)` | Set volume, clamped to `[0, 1]` |

### `KnowledgePanel`

SpinStep `Node` with multi-page content display.

| Method | Returns | Description |
|---|---|---|
| `show()` | `None` | Make panel visible |
| `hide()` | `None` | Hide panel |
| `advance_page()` | `bool` | Advance to next page; returns `False` at last page |
| `current_content` | `PanelPage \| None` | Current page |

---

## `vrspin.plaza` — Virtual Plaza

```python
from vrspin import VirtualPlaza, PlazaEvent
```

### `VirtualPlaza`

Full simulation engine: builds a SpinStep `Node` tree and drives attention-cone
mechanics each tick.

| Method | Returns | Description |
|---|---|---|
| `tick(user)` | `list[PlazaEvent]` | Advance one simulation step |
| `get_object(name)` | `InteractiveObject \| None` | Look up interactive object |
| `get_npc(name)` | `NPC \| None` | Look up NPC |

### `PlazaEvent`

```python
@dataclass
class PlazaEvent:
    tick: int
    modality: str  # 'visual', 'audio', 'haptic', 'npc', 'knowledge', 'system'
    source: str
    message: str
```

---

## Utility Functions

```python
from vrspin import forward_vector_from_quaternion, slerp
```

### `forward_vector_from_quaternion(q) → np.ndarray`

Extract the forward (look) direction from a quaternion.
Re-exported from `spinstep.utils`.  Uses the `-Z` convention: identity
quaternion `[0, 0, 0, 1]` returns `[0, 0, -1]`.

### `direction_to_quaternion(direction) → np.ndarray`

Convert a 3D direction vector to an orientation quaternion.
Re-exported from `spinstep.utils`.

```python
q = direction_to_quaternion([0, 0, -1])  # identity quaternion
```

### `angle_between_directions(d1, d2) → float`

Angular distance in radians between two direction vectors.
Re-exported from `spinstep.utils`.

```python
angle = angle_between_directions([1, 0, 0], [0, 1, 0])  # π/2
```

### `slerp(q1, q2, t) → np.ndarray`

Spherical linear interpolation between two quaternions.

```python
mid = slerp([0, 0, 0, 1], [0, 0.707, 0, 0.707], 0.5)
```

---

## Quaternion Convention

All quaternions throughout VRSpin use the `[x, y, z, w]` convention,
matching SpinStep and `scipy.spatial.transform.Rotation`.

All angles are in **radians** unless otherwise noted.
