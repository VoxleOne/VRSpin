/**
 * @file VRSpinNative.h
 * @brief Unreal Engine C++ wrapper for the VRSpin native C library.
 *
 * Drop this header into your Unreal project's Source/ folder and link
 * against the compiled VRSpin shared library.
 *
 * Build setup (in your module's Build.cs):
 * @code
 *   PublicAdditionalLibraries.Add(Path.Combine(PluginDir, "libvrspin_native.so"));
 *   // or on Windows:
 *   PublicAdditionalLibraries.Add(Path.Combine(PluginDir, "vrspin_native.lib"));
 *   PublicDelayLoadDLLs.Add("vrspin_native.dll");
 * @endcode
 *
 * Usage:
 * @code
 *   #include "VRSpinNative.h"
 *
 *   FVRSpinCone Cone(PlayerQuat, FMath::DegreesToRadians(45.0), EVRSpinFalloff::Linear);
 *   if (Cone.Contains(EntityQuat))
 *   {
 *       float Strength = Cone.Attenuation(EntityQuat);
 *       // apply highlight ...
 *   }
 * @endcode
 */

#pragma once

#include "CoreMinimal.h"

// Pull in the C API header.
// Adjust this path based on your project layout.
extern "C"
{
#include "vrspin_native.h"
}

/**
 * Falloff mode for attention cones.
 */
enum class EVRSpinFalloff : int32
{
    None   = VRSPIN_FALLOFF_NONE,
    Linear = VRSPIN_FALLOFF_LINEAR,
    Cosine = VRSPIN_FALLOFF_COSINE,
};

/**
 * Helper: convert an FQuat to a double[4] in VRSpin's [x,y,z,w] order.
 *
 * Note: Unreal's FQuat stores (X, Y, Z, W) — same convention.
 */
inline void QuatToVRSpin(const FQuat& Q, double Out[4])
{
    Out[0] = Q.X;
    Out[1] = Q.Y;
    Out[2] = Q.Z;
    Out[3] = Q.W;
}

/** Convert a VRSpin double[4] back to an FQuat. */
inline FQuat VRSpinToQuat(const double In[4])
{
    return FQuat(In[0], In[1], In[2], In[3]);
}

// -----------------------------------------------------------------------
// Utility free functions
// -----------------------------------------------------------------------

/** Angular distance (radians) between two quaternions. */
inline double VRSpinQuatDistance(const FQuat& A, const FQuat& B)
{
    double a[4], b[4];
    QuatToVRSpin(A, a);
    QuatToVRSpin(B, b);
    return vrspin_quat_distance(a, b);
}

/** Extract the forward direction (-Z convention) from a quaternion. */
inline FVector VRSpinForwardVector(const FQuat& Q)
{
    double q[4], v[3];
    QuatToVRSpin(Q, q);
    vrspin_forward_vector(q, v);
    return FVector(v[0], v[1], v[2]);
}

/** Spherical linear interpolation between two quaternions. */
inline FQuat VRSpinSlerp(const FQuat& A, const FQuat& B, double T)
{
    double a[4], b[4], out[4];
    QuatToVRSpin(A, a);
    QuatToVRSpin(B, b);
    vrspin_slerp(a, b, T, out);
    return VRSpinToQuat(out);
}

// -----------------------------------------------------------------------
// RAII wrapper around VRSpinCone
// -----------------------------------------------------------------------

/**
 * C++ RAII wrapper for a native VRSpin attention cone.
 *
 * Automatically creates and destroys the underlying C handle.
 */
class FVRSpinCone
{
public:
    /**
     * Construct a new attention cone.
     *
     * @param Orientation  Centre orientation.
     * @param HalfAngle    Half-aperture in radians.
     * @param Falloff      Attenuation curve.
     */
    FVRSpinCone(const FQuat& Orientation, double HalfAngle,
                EVRSpinFalloff Falloff = EVRSpinFalloff::None)
    {
        double q[4];
        QuatToVRSpin(Orientation, q);
        Handle = vrspin_cone_create(q, HalfAngle, static_cast<int>(Falloff));
        check(Handle != nullptr);
    }

    ~FVRSpinCone()
    {
        if (Handle)
        {
            vrspin_cone_destroy(Handle);
            Handle = nullptr;
        }
    }

    // Non-copyable, movable.
    FVRSpinCone(const FVRSpinCone&) = delete;
    FVRSpinCone& operator=(const FVRSpinCone&) = delete;

    FVRSpinCone(FVRSpinCone&& Other) noexcept : Handle(Other.Handle)
    {
        Other.Handle = nullptr;
    }

    FVRSpinCone& operator=(FVRSpinCone&& Other) noexcept
    {
        if (this != &Other)
        {
            if (Handle) vrspin_cone_destroy(Handle);
            Handle = Other.Handle;
            Other.Handle = nullptr;
        }
        return *this;
    }

    /** Update the cone's pointing direction. */
    void UpdateOrigin(const FQuat& NewQuat)
    {
        double q[4];
        QuatToVRSpin(NewQuat, q);
        vrspin_cone_update_origin(Handle, q);
    }

    /** Test whether a target orientation falls inside the cone. */
    bool Contains(const FQuat& Target) const
    {
        double q[4];
        QuatToVRSpin(Target, q);
        return vrspin_cone_contains(Handle, q) != 0;
    }

    /** Attention strength in [0, 1] for a target orientation. */
    double Attenuation(const FQuat& Target) const
    {
        double q[4];
        QuatToVRSpin(Target, q);
        return vrspin_cone_attenuation(Handle, q);
    }

    /**
     * Batch membership test.
     *
     * @param Targets  Array of quaternions to test.
     * @return         Array of booleans (true = inside cone).
     */
    TArray<bool> QueryBatch(const TArray<FQuat>& Targets) const
    {
        int32 N = Targets.Num();
        TArray<double> Flat;
        Flat.SetNumUninitialized(N * 4);
        for (int32 i = 0; i < N; i++)
            QuatToVRSpin(Targets[i], &Flat[i * 4]);

        TArray<int32> IntResults;
        IntResults.SetNumZeroed(N);
        vrspin_cone_query_batch(Handle, Flat.GetData(), N, IntResults.GetData());

        TArray<bool> Results;
        Results.SetNumUninitialized(N);
        for (int32 i = 0; i < N; i++)
            Results[i] = IntResults[i] != 0;
        return Results;
    }

    /**
     * Batch attenuation query.
     *
     * @param Targets  Array of quaternions.
     * @return         Array of doubles in [0, 1].
     */
    TArray<double> QueryBatchAttenuation(const TArray<FQuat>& Targets) const
    {
        int32 N = Targets.Num();
        TArray<double> Flat;
        Flat.SetNumUninitialized(N * 4);
        for (int32 i = 0; i < N; i++)
            QuatToVRSpin(Targets[i], &Flat[i * 4]);

        TArray<double> Results;
        Results.SetNumZeroed(N);
        vrspin_cone_query_batch_attenuation(Handle, Flat.GetData(), N,
                                             Results.GetData());
        return Results;
    }

private:
    VRSpinCone* Handle = nullptr;
};

// -----------------------------------------------------------------------
// Frame processing
// -----------------------------------------------------------------------

/** Result of a full-frame attention query. */
struct FVRSpinFrameResult
{
    TArray<double> VisualStrengths;
    TArray<double> AudioGains;
    TArray<bool>   VisualAttended;
    int32          AttendedCount;
};

/**
 * Process a full attention frame for N entities.
 *
 * @param UserQuat             Head orientation.
 * @param EntityOrientations   Per-entity orientations.
 * @param VisualHalfAngle      Visual cone half-angle (radians).
 * @param AudioHalfAngle       Audio cone half-angle (radians).
 * @return                     Per-entity attention results.
 */
inline FVRSpinFrameResult VRSpinProcessFrame(
    const FQuat& UserQuat,
    const TArray<FQuat>& EntityOrientations,
    double VisualHalfAngle,
    double AudioHalfAngle)
{
    int32 N = EntityOrientations.Num();
    double uq[4];
    QuatToVRSpin(UserQuat, uq);

    TArray<double> Flat;
    Flat.SetNumUninitialized(N * 4);
    for (int32 i = 0; i < N; i++)
        QuatToVRSpin(EntityOrientations[i], &Flat[i * 4]);

    FVRSpinFrameResult Result;
    Result.VisualStrengths.SetNumZeroed(N);
    Result.AudioGains.SetNumZeroed(N);
    Result.VisualAttended.SetNumUninitialized(N);

    TArray<int32> IntAttended;
    IntAttended.SetNumZeroed(N);

    Result.AttendedCount = vrspin_process_frame(
        uq, Flat.GetData(), N,
        VisualHalfAngle, AudioHalfAngle,
        Result.VisualStrengths.GetData(),
        Result.AudioGains.GetData(),
        IntAttended.GetData());

    for (int32 i = 0; i < N; i++)
        Result.VisualAttended[i] = IntAttended[i] != 0;

    return Result;
}
