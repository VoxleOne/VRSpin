// VRSpinNative.cs — Unity C# bindings for the VRSpin native C library.
//
// Loads the compiled VRSpin shared library via P/Invoke and provides
// managed wrappers for attention-cone queries, quaternion utilities,
// and full-frame processing.
//
// Usage:
//   1. Copy vrspin_native.dll (Windows) or libvrspin_native.so (Linux)
//      into Assets/Plugins/.
//   2. Attach VRSpinBridge (below) to a GameObject in your scene.
//   3. Call VRSpinNative.ConeContains() etc. from any script.
//
// See also: examples/unity/SpinStepClient.cs for the WebSocket variant.

using System;
using System.Runtime.InteropServices;
using UnityEngine;

/// <summary>
/// Low-level P/Invoke declarations for the VRSpin native C library.
/// All quaternions use [x, y, z, w] order (same as Unity's Quaternion).
/// All angles are in radians.
/// </summary>
public static class VRSpinNative
{
    // ----------------------------------------------------------------
    // Library name — adjust per platform if needed.
    // Unity resolves this to vrspin_native.dll / libvrspin_native.so
    // from Assets/Plugins/.
    // ----------------------------------------------------------------
    private const string LibName = "vrspin_native";

    // ----------------------------------------------------------------
    // Version
    // ----------------------------------------------------------------

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr vrspin_version();

    /// <summary>Return the native library version string.</summary>
    public static string Version()
    {
        return Marshal.PtrToStringAnsi(vrspin_version());
    }

    // ----------------------------------------------------------------
    // Quaternion utilities
    // ----------------------------------------------------------------

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern double vrspin_quat_distance(double[] q1, double[] q2);

    /// <summary>
    /// Angular distance (radians) between two quaternions.
    /// </summary>
    public static double QuatDistance(Quaternion a, Quaternion b)
    {
        return vrspin_quat_distance(ToArray(a), ToArray(b));
    }

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void vrspin_forward_vector(double[] q, double[] outVec);

    /// <summary>
    /// Extract the forward direction (−Z convention) from a quaternion.
    /// </summary>
    public static Vector3 ForwardVector(Quaternion q)
    {
        var qArr = ToArray(q);
        var v = new double[3];
        vrspin_forward_vector(qArr, v);
        return new Vector3((float)v[0], (float)v[1], (float)v[2]);
    }

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void vrspin_slerp(double[] q1, double[] q2, double t, double[] outQ);

    /// <summary>
    /// Spherical linear interpolation between two quaternions.
    /// </summary>
    public static Quaternion Slerp(Quaternion a, Quaternion b, float t)
    {
        var outQ = new double[4];
        vrspin_slerp(ToArray(a), ToArray(b), t, outQ);
        return FromArray(outQ);
    }

    // ----------------------------------------------------------------
    // Attention cone
    // ----------------------------------------------------------------

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr vrspin_cone_create(double[] orientation,
                                                     double halfAngle,
                                                     int falloff);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void vrspin_cone_destroy(IntPtr cone);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void vrspin_cone_update_origin(IntPtr cone, double[] newQuat);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int vrspin_cone_contains(IntPtr cone, double[] target);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern double vrspin_cone_attenuation(IntPtr cone, double[] target);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void vrspin_cone_query_batch(IntPtr cone,
                                                        double[] quats,
                                                        int n,
                                                        int[] results);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void vrspin_cone_query_batch_attenuation(IntPtr cone,
                                                                    double[] quats,
                                                                    int n,
                                                                    double[] results);

    // ----------------------------------------------------------------
    // Frame processing
    // ----------------------------------------------------------------

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int vrspin_process_frame(double[] userQuat,
                                                    double[] entityOrientations,
                                                    int numEntities,
                                                    double visualHalfAngle,
                                                    double audioHalfAngle,
                                                    double[] outVisualStrengths,
                                                    double[] outAudioGains,
                                                    int[] outVisualAttended);

    // ----------------------------------------------------------------
    // Managed wrappers
    // ----------------------------------------------------------------

    /// <summary>Falloff modes matching the C VRSPIN_FALLOFF_* constants.</summary>
    public enum Falloff { None = 0, Linear = 1, Cosine = 2 }

    /// <summary>
    /// Managed wrapper around a native VRSpin attention cone.
    /// Dispose when no longer needed to free native memory.
    /// </summary>
    public class AttentionCone : IDisposable
    {
        private IntPtr _handle;

        public AttentionCone(Quaternion orientation, float halfAngleRad,
                             Falloff falloff = Falloff.None)
        {
            _handle = vrspin_cone_create(ToArray(orientation),
                                          halfAngleRad, (int)falloff);
            if (_handle == IntPtr.Zero)
                throw new OutOfMemoryException("Failed to create native cone");
        }

        public void UpdateOrigin(Quaternion q)
        {
            vrspin_cone_update_origin(_handle, ToArray(q));
        }

        public bool Contains(Quaternion target)
        {
            return vrspin_cone_contains(_handle, ToArray(target)) != 0;
        }

        public double Attenuation(Quaternion target)
        {
            return vrspin_cone_attenuation(_handle, ToArray(target));
        }

        public bool[] QueryBatch(Quaternion[] targets)
        {
            int n = targets.Length;
            var flat = new double[n * 4];
            for (int i = 0; i < n; i++)
            {
                flat[i * 4 + 0] = targets[i].x;
                flat[i * 4 + 1] = targets[i].y;
                flat[i * 4 + 2] = targets[i].z;
                flat[i * 4 + 3] = targets[i].w;
            }
            var results = new int[n];
            vrspin_cone_query_batch(_handle, flat, n, results);
            var bools = new bool[n];
            for (int i = 0; i < n; i++) bools[i] = results[i] != 0;
            return bools;
        }

        public double[] QueryBatchAttenuation(Quaternion[] targets)
        {
            int n = targets.Length;
            var flat = new double[n * 4];
            for (int i = 0; i < n; i++)
            {
                flat[i * 4 + 0] = targets[i].x;
                flat[i * 4 + 1] = targets[i].y;
                flat[i * 4 + 2] = targets[i].z;
                flat[i * 4 + 3] = targets[i].w;
            }
            var results = new double[n];
            vrspin_cone_query_batch_attenuation(_handle, flat, n, results);
            return results;
        }

        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
            {
                vrspin_cone_destroy(_handle);
                _handle = IntPtr.Zero;
            }
        }

        ~AttentionCone() { Dispose(); }
    }

    /// <summary>
    /// Result of a single-frame attention query.
    /// </summary>
    public struct FrameResult
    {
        public double[] VisualStrengths;
        public double[] AudioGains;
        public bool[] VisualAttended;
        public int AttendedCount;
    }

    /// <summary>
    /// Process a full attention frame (visual + audio) for N entities.
    /// This is the recommended entry point for per-frame VR integration.
    /// </summary>
    public static FrameResult ProcessFrame(Quaternion userQuat,
                                            Quaternion[] entityOrientations,
                                            float visualHalfAngle,
                                            float audioHalfAngle)
    {
        int n = entityOrientations.Length;
        var flat = new double[n * 4];
        for (int i = 0; i < n; i++)
        {
            flat[i * 4 + 0] = entityOrientations[i].x;
            flat[i * 4 + 1] = entityOrientations[i].y;
            flat[i * 4 + 2] = entityOrientations[i].z;
            flat[i * 4 + 3] = entityOrientations[i].w;
        }

        var visStr = new double[n];
        var audGain = new double[n];
        var visAtt = new int[n];

        int count = vrspin_process_frame(ToArray(userQuat), flat, n,
                                          visualHalfAngle, audioHalfAngle,
                                          visStr, audGain, visAtt);

        var bools = new bool[n];
        for (int i = 0; i < n; i++) bools[i] = visAtt[i] != 0;

        return new FrameResult
        {
            VisualStrengths = visStr,
            AudioGains = audGain,
            VisualAttended = bools,
            AttendedCount = count,
        };
    }

    // ----------------------------------------------------------------
    // Internal helpers
    // ----------------------------------------------------------------

    private static double[] ToArray(Quaternion q)
    {
        // Unity Quaternion stores (x, y, z, w) — same as VRSpin convention.
        return new double[] { q.x, q.y, q.z, q.w };
    }

    private static Quaternion FromArray(double[] arr)
    {
        return new Quaternion((float)arr[0], (float)arr[1],
                              (float)arr[2], (float)arr[3]);
    }
}


/// <summary>
/// Unity MonoBehaviour that demonstrates native VRSpin integration.
///
/// Attach to a GameObject in your scene.  Each frame it:
///   1. Reads the main camera's rotation (head quaternion).
///   2. Gathers all tagged scene entities.
///   3. Calls VRSpinNative.ProcessFrame() for sub-millisecond
///      attention computation.
///   4. Highlights attended objects with a glow effect.
///
/// This replaces the WebSocket bridge for production deployments.
/// </summary>
public class VRSpinBridge : MonoBehaviour
{
    [Tooltip("Visual attention half-angle in degrees")]
    public float visualHalfAngleDeg = 45f;

    [Tooltip("Audio attention half-angle in degrees")]
    public float audioHalfAngleDeg = 90f;

    [Tooltip("Tag used to find scene entities")]
    public string entityTag = "VRSpinEntity";

    private GameObject[] _entities;

    void Start()
    {
        Debug.Log($"VRSpin Native v{VRSpinNative.Version()} loaded");
        _entities = GameObject.FindGameObjectsWithTag(entityTag);
    }

    void Update()
    {
        if (_entities == null || _entities.Length == 0) return;

        Quaternion headQuat = Camera.main.transform.rotation;
        int n = _entities.Length;
        var orientations = new Quaternion[n];
        for (int i = 0; i < n; i++)
            orientations[i] = _entities[i].transform.rotation;

        float visHalf = visualHalfAngleDeg * Mathf.Deg2Rad;
        float audHalf = audioHalfAngleDeg * Mathf.Deg2Rad;

        var result = VRSpinNative.ProcessFrame(headQuat, orientations,
                                                visHalf, audHalf);

        for (int i = 0; i < n; i++)
        {
            var renderer = _entities[i].GetComponent<Renderer>();
            if (renderer != null)
            {
                renderer.material.SetFloat("_GlowStrength",
                    (float)result.VisualStrengths[i]);
            }

            var audioSrc = _entities[i].GetComponent<AudioSource>();
            if (audioSrc != null)
            {
                audioSrc.volume = (float)result.AudioGains[i];
            }
        }
    }
}
