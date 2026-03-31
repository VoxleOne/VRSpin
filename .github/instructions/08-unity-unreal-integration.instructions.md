## Unity / Unreal Integration
###  Integration Architecture

VRSpin runs as a Python service that communicates with the VR engine via a lightweight bridge. Two integration patterns are supported:

* Option A: WebSocket Bridge (Recommended for Prototyping)

Unity/Unreal  ←→  WebSocket  ←→  Python (SpinStep + FastAPI/asyncio)

    VR engine sends headset quaternion + entity states as JSON every frame
    Python computes attention results, NPC updates
    Results sent back as JSON (highlighted entities, NPC rotations, audio gains)
    Latency: ~2–5ms on localhost

* Option B: C Extension / Native Plugin (Production)

Unity/Unreal  ←→  C API (pybind11 or ctypes)  ←→  SpinStep

    VRSpin compiled as a shared library via pybind11
    Direct function calls from C# (Unity) or C++ (Unreal)
    Latency: <0.1ms

### WebSocket Message Protocol
Client → Server (VR Engine → VRSpin)

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
    }
  ]
}

Server → Client (VRSpin → VR Engine)

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
    {"id": "fountain_audio", "gain": 0.72}
  ]
}

### Unity C# Client (Sketch)

// Unity MonoBehaviour — sends head orientation to SpinStep each frame
void Update() {
    Quaternion headQuat = Camera.main.transform.rotation;
    // Convert Unity's (x,y,z,w) to SpinStep's [x,y,z,w] (same order)
    string msg = JsonUtility.ToJson(new FrameUpdate {
        head_quaternion = new float[] {
            headQuat.x, headQuat.y, headQuat.z, headQuat.w
        }
    });
    websocket.Send(msg);
}

// Handle response
void OnAttentionResult(AttentionResult result) {
    foreach (var entity in result.attended_entities) {
        GameObject obj = scene.Find(entity.id);
        obj.GetComponent<Renderer>().material.SetFloat("_GlowStrength",
            entity.attention_strength);
    }
}
