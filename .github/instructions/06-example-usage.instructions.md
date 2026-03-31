## Example Usage
### Basic: User Looks at an Object

import numpy as np
from vrspin import AttentionCone
from spinstep.utils import quaternion_from_euler

#### User is looking slightly to the right (30° yaw)
user_quat = quaternion_from_euler([30, 0, 0], order='yxz', degrees=True)

#### Attention cone: 45° half-angle (90° total field of view)
cone = AttentionCone(user_quat, half_angle=np.radians(45))

#### Fountain is at 25° to the right of center
fountain_quat = quaternion_from_euler([25, 5, 0], order='yxz', degrees=True)

print(cone.contains(fountain_quat))       # True — fountain is in view
print(cone.attenuation(fountain_quat))    # 0.89 — close to center of cone

### Scene: Full Plaza with Multiple Entities

from vrspin import SceneEntity, AttentionManager, AttentionCone
from spinstep.utils import quaternion_from_euler
import numpy as np

#### Build scene
fountain = SceneEntity(
    name="fountain",
    orientation=quaternion_from_euler([0, 0, 0]),
    position=[5.0, 0.0, 3.0],
    entity_type="object",
)
npc_vendor = SceneEntity(
    name="vendor",
    orientation=quaternion_from_euler([90, 0, 0]),
    position=[-3.0, 0.0, 2.0],
    entity_type="npc",
)
art_panel = SceneEntity(
    name="vr_art_panel",
    orientation=quaternion_from_euler([180, 0, 0]),
    position=[0.0, 2.0, -4.0],
    entity_type="panel",
    metadata={"content": "VR Art: A New Medium"},
)

#### Create attention manager
manager = AttentionManager([fountain, npc_vendor, art_panel])

#### Simulate user looking toward the fountain
user_quat = quaternion_from_euler([5, -2, 0], order='yxz', degrees=True)
result = manager.update(user_quat, cone_half_angle=np.radians(45))

for entity, strength in result.attended:
    print(f"  {entity.name}: attention={strength:.2f}")
    # fountain: attention=0.94

### NPC: Vendor Notices the User

from vrspin.npc import NPCAttentionAgent
from spinstep.utils import quaternion_from_euler
import numpy as np

#### NPC vendor with 80° perception cone
npc_agent = NPCAttentionAgent(
    entity=npc_vendor,
    perception_half_angle=np.radians(40),
    turn_speed=0.15,
)

#### User walks into the vendor's perception cone
user_quat = quaternion_from_euler([85, 0, 0], order='yxz', degrees=True)

if npc_agent.is_aware_of(user_quat):
    print("Vendor notices the user!")
    # Over several frames, vendor smoothly turns toward user
    for frame in range(10):
        npc_agent.update(targets=[user_quat], dt=1/60)
    print(f"Vendor now facing: {npc_vendor.orientation}")

### Multi-Head: Visual + Audio + Haptic Cones

from vrspin import AttentionCone
from vrspin.multihead import MultiHeadAttention
from spinstep.utils import quaternion_from_euler
import numpy as np

user_quat = quaternion_from_euler([30, 0, 0], order='yxz', degrees=True)

multi = MultiHeadAttention({
    'visual': AttentionCone(user_quat, half_angle=np.radians(45)),
    'audio':  AttentionCone(user_quat, half_angle=np.radians(90)),  # wider
    'haptic': AttentionCone(user_quat, half_angle=np.radians(20)),  # narrower
})

entities = [fountain, npc_vendor, art_panel]
results = multi.update(user_quat, entities)

print("Visual:", [e.name for e, _ in results['visual']])
print("Audio:",  [e.name for e, _ in results['audio']])   # may include more
print("Haptic:", [e.name for e, _ in results['haptic']])   # only very close

#### Merge: entities attended by ANY modality
all_attended = multi.merge_results(strategy='union')

### Audio Focus: Spatial Audio Gain Based on Cone

from vrspin import AttentionCone
import numpy as np

#### Audio cone — wider than visual
audio_cone = AttentionCone(user_quat, half_angle=np.radians(90), falloff='cosine')

#### Compute per-source audio gain
audio_sources = [fountain, npc_vendor]  # both have spatial audio
for source in audio_sources:
    gain = audio_cone.attenuation(source.orientation)
    # gain: 0.0 (outside cone) to 1.0 (dead center)
    print(f"{source.name} audio gain: {gain:.2f}")
    # → Pass gain to spatial audio engine