# VRSpin WebSocket Bridge Server

Connects a Unity or Unreal Engine client to the VRSpin Python simulation layer
via a lightweight JSON-over-WebSocket protocol.

---

## Quick Start

```bash
# Install bridge dependency
pip install websockets

# Start the bridge (default: localhost:8765)
python examples/vr_bridge_server.py

# Custom host / port
python examples/vr_bridge_server.py --host 0.0.0.0 --port 9000
```

---

## Protocol

### Client → Server: `frame_update`

Sent by the VR engine every frame (or on every head-pose change).

```json
{
  "type": "frame_update",
  "timestamp": 1616700000.123,
  "user": {
    "head_quaternion": [0.0, 0.1, 0.0, 0.995],
    "position": [1.0, 1.7, 3.0]
  },
  "entities": [
    {
      "id": "fountain",
      "orientation": [0.0, 0.0, 0.0, 1.0],
      "position": [5.0, 0.0, 3.0],
      "type": "object"
    },
    {
      "id": "vendor",
      "orientation": [0.0, 0.0, 0.0, 1.0],
      "position": [-3.0, 0.0, 2.0],
      "type": "npc"
    }
  ]
}
```

**Quaternion convention:** `[x, y, z, w]` — same as Unity's
`new float[] { q.x, q.y, q.z, q.w }`.

### Server → Client: `attention_result`

Returned for every `frame_update`.

```json
{
  "type": "attention_result",
  "timestamp": 1616700000.125,
  "attended_entities": [
    {"id": "fountain", "attention_strength": 0.89, "highlight": true}
  ],
  "npc_updates": [
    {"id": "vendor", "new_orientation": [0.05, 0.1, 0.0, 0.994], "state": "aware"}
  ],
  "audio_gains": [
    {"id": "fountain", "gain": 0.72}
  ]
}
```

| Field | Description |
|---|---|
| `attended_entities` | Entities inside the visual cone (half-angle 45°), with `attention_strength` in `[0, 1]` |
| `npc_updates` | Current NPC state (`"aware"` or `"idle"`) |
| `audio_gains` | Per-entity audio gain in `[0, 1]` from audio cone (half-angle 90°, cosine falloff) |

### Error Response

```json
{"error": "invalid JSON"}
{"error": "unknown message type: ping"}
```

---

## Unity C# Integration

```csharp
using UnityEngine;
using NativeWebSocket;  // or any WebSocket library

public class VRSpinClient : MonoBehaviour
{
    WebSocket websocket;

    async void Start()
    {
        websocket = new WebSocket("ws://localhost:8765");
        websocket.OnMessage += OnMessage;
        await websocket.Connect();
    }

    void Update()
    {
        // Send head orientation every frame
        Quaternion q = Camera.main.transform.rotation;
        string msg = JsonUtility.ToJson(new FrameUpdate {
            type = "frame_update",
            timestamp = Time.time,
            user = new UserPose {
                head_quaternion = new float[] { q.x, q.y, q.z, q.w },
                position = new float[] {
                    transform.position.x,
                    transform.position.y,
                    transform.position.z,
                }
            }
        });
        websocket.SendText(msg);
        websocket.DispatchMessageQueue();
    }

    void OnMessage(byte[] bytes)
    {
        AttentionResult result = JsonUtility.FromJson<AttentionResult>(
            System.Text.Encoding.UTF8.GetString(bytes)
        );
        foreach (var entity in result.attended_entities)
        {
            GameObject obj = GameObject.Find(entity.id);
            if (obj != null)
            {
                obj.GetComponent<Renderer>().material
                   .SetFloat("_GlowStrength", entity.attention_strength);
            }
        }
        foreach (var gain in result.audio_gains)
        {
            AudioSource src = GameObject.Find(gain.id)?.GetComponent<AudioSource>();
            if (src != null) src.volume = gain.gain;
        }
    }
}
```

See `examples/unity/SpinStepClient.cs` for the full sketch.

---

## Latency

| Mode | Typical round-trip |
|---|---|
| WebSocket on localhost | ~2–5 ms |
| WebSocket over LAN | ~5–15 ms |
| C-extension (pybind11) | < 0.1 ms |

The WebSocket bridge is recommended for prototyping.
For production, compile VRSpin as a native shared library via `pybind11`.

---

## Dependencies

| Package | Purpose |
|---|---|
| `websockets` | Async WebSocket server (`pip install websockets`) |
| `numpy` | Quaternion arrays |
| `scipy` | SLERP, Rotation |
| `spinstep` | SpinStep quaternion primitives |

VRSpin itself (`pip install -e .`) installs `numpy` and `scipy` automatically.
Only `websockets` must be installed separately.

---

## See Also

- [VR Demo Guide](../docs/10-vr-demo.md)
- [API Reference](../docs/09-api-reference.md)
- [Unity client sketch](unity/SpinStepClient.cs)
