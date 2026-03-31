// SpinStepClient.cs — Unity MonoBehaviour sketch for VRSpin WebSocket bridge.
//
// Attach this component to a GameObject in your Unity scene.  It sends the
// headset's quaternion orientation to VRSpin every frame via WebSocket and
// applies the returned attention results (entity highlights, audio gains,
// NPC updates) to the scene.
//
// Requirements:
//   - NativeWebSocket Unity package (https://github.com/endel/NativeWebSocket)
//   - VRSpin bridge server running: python examples/vr_bridge_server.py
//
// Usage:
//   1. Start the Python server.
//   2. In Unity, set the "ServerUrl" field to ws://localhost:8765 (default).
//   3. Press Play.

using System;
using System.Collections.Generic;
using UnityEngine;
using NativeWebSocket;  // https://github.com/endel/NativeWebSocket

public class SpinStepClient : MonoBehaviour
{
    [Header("Bridge Connection")]
    public string ServerUrl = "ws://localhost:8765";

    private WebSocket _ws;

    // ------------------------------------------------------------------
    // Data classes matching the VRSpin JSON protocol
    // ------------------------------------------------------------------

    [Serializable]
    public class FrameUpdate
    {
        public string type = "frame_update";
        public double timestamp;
        public UserPose user;
        public List<EntityData> entities;
    }

    [Serializable]
    public class UserPose
    {
        public float[] head_quaternion;
        public float[] position;
    }

    [Serializable]
    public class EntityData
    {
        public string id;
        public float[] orientation;
        public float[] position;
        public string type;
    }

    [Serializable]
    public class AttentionResult
    {
        public string type;
        public double timestamp;
        public List<AttendedEntity> attended_entities;
        public List<NPCUpdate> npc_updates;
        public List<AudioGain> audio_gains;
    }

    [Serializable]
    public class AttendedEntity
    {
        public string id;
        public float attention_strength;
        public bool highlight;
    }

    [Serializable]
    public class NPCUpdate
    {
        public string id;
        public float[] new_orientation;
        public string state;
    }

    [Serializable]
    public class AudioGain
    {
        public string id;
        public float gain;
    }

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    async void Start()
    {
        _ws = new WebSocket(ServerUrl);
        _ws.OnMessage += OnMessage;
        _ws.OnError += (err) => Debug.LogError($"[SpinStep] WS error: {err}");
        _ws.OnClose += (code) => Debug.Log($"[SpinStep] WS closed: {code}");
        await _ws.Connect();
    }

    void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        _ws?.DispatchMessageQueue();
#endif

        if (_ws?.State != WebSocketState.Open) return;

        Quaternion headQuat = Camera.main.transform.rotation;
        var msg = new FrameUpdate
        {
            timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0,
            user = new UserPose
            {
                // Unity uses (x, y, z, w) — same order as SpinStep
                head_quaternion = new float[]
                {
                    headQuat.x, headQuat.y, headQuat.z, headQuat.w
                },
                position = new float[]
                {
                    Camera.main.transform.position.x,
                    Camera.main.transform.position.y,
                    Camera.main.transform.position.z,
                },
            },
            entities = CollectSceneEntities(),
        };

        string json = JsonUtility.ToJson(msg);
        _ws.SendText(json);
    }

    async void OnApplicationQuit()
    {
        if (_ws != null) await _ws.Close();
    }

    // ------------------------------------------------------------------
    // Response handling
    // ------------------------------------------------------------------

    void OnMessage(byte[] data)
    {
        string json = System.Text.Encoding.UTF8.GetString(data);
        var result = JsonUtility.FromJson<AttentionResult>(json);
        if (result == null || result.type != "attention_result") return;

        // Apply highlights
        foreach (var entity in result.attended_entities)
        {
            GameObject obj = GameObject.Find(entity.id);
            if (obj == null) continue;
            var renderer = obj.GetComponent<Renderer>();
            if (renderer != null)
            {
                renderer.material.SetFloat("_GlowStrength",
                    entity.attention_strength);
            }
        }

        // Apply NPC orientation updates
        foreach (var npc in result.npc_updates)
        {
            GameObject obj = GameObject.Find(npc.id);
            if (obj == null) continue;
            if (npc.new_orientation != null && npc.new_orientation.Length == 4)
            {
                obj.transform.rotation = new Quaternion(
                    npc.new_orientation[0], npc.new_orientation[1],
                    npc.new_orientation[2], npc.new_orientation[3]);
            }
        }

        // Apply audio gains
        foreach (var ag in result.audio_gains)
        {
            GameObject obj = GameObject.Find(ag.id);
            if (obj == null) continue;
            var src = obj.GetComponent<AudioSource>();
            if (src != null) src.volume = ag.gain;
        }
    }

    // ------------------------------------------------------------------
    // Scene entity collection (customise for your project)
    // ------------------------------------------------------------------

    List<EntityData> CollectSceneEntities()
    {
        // Placeholder: collect tagged GameObjects.
        // In a real project, iterate over your entity registry.
        var list = new List<EntityData>();
        foreach (var go in GameObject.FindGameObjectsWithTag("VRSpinEntity"))
        {
            var q = go.transform.rotation;
            var p = go.transform.position;
            list.Add(new EntityData
            {
                id = go.name,
                orientation = new float[] { q.x, q.y, q.z, q.w },
                position = new float[] { p.x, p.y, p.z },
                type = "object",
            });
        }
        return list;
    }
}
