/**
 * @file vrspin_native.h
 * @brief Native C API for VRSpin attention-cone computations.
 *
 * This header defines the public C interface for VRSpin's core
 * orientation-based attention system.  The library can be loaded by:
 *
 *   - Python via ``ctypes`` (see ``vrspin/native_bridge.py``)
 *   - Unity via C# ``DllImport`` (P/Invoke)
 *   - Unreal via C++ ``dlopen`` / ``LoadLibrary``
 *
 * All quaternions use the ``[x, y, z, w]`` convention.
 * All angles are in radians.
 */

#ifndef VRSPIN_NATIVE_H
#define VRSPIN_NATIVE_H

#include <stdint.h>

/* ------------------------------------------------------------------ */
/* Platform export / visibility                                        */
/* ------------------------------------------------------------------ */

#ifdef _WIN32
  #ifdef VRSPIN_BUILD_DLL
    #define VRSPIN_API __declspec(dllexport)
  #else
    #define VRSPIN_API __declspec(dllimport)
  #endif
#else
  #define VRSPIN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Falloff constants                                                    */
/* ------------------------------------------------------------------ */

/** Step function: 1.0 inside the cone, 0.0 outside. */
#define VRSPIN_FALLOFF_NONE   0
/** Linear decay from 1.0 at centre to 0.0 at the edge. */
#define VRSPIN_FALLOFF_LINEAR 1
/** Cosine decay (smoother roll-off at the edge). */
#define VRSPIN_FALLOFF_COSINE 2

/* ------------------------------------------------------------------ */
/* Quaternion utilities                                                 */
/* ------------------------------------------------------------------ */

/**
 * Normalise a quaternion in-place.
 *
 * @param q  Quaternion [x, y, z, w] — modified in-place.
 */
VRSPIN_API void vrspin_quat_normalize(double q[4]);

/**
 * Compute the geodesic angular distance between two unit quaternions.
 *
 * @param q1  First quaternion [x, y, z, w].
 * @param q2  Second quaternion [x, y, z, w].
 * @return    Angle in radians in [0, pi].
 */
VRSPIN_API double vrspin_quat_distance(const double q1[4], const double q2[4]);

/**
 * Extract the forward direction vector from a quaternion.
 *
 * Uses the -Z convention: identity quaternion [0,0,0,1] yields [0,0,-1].
 *
 * @param q    Quaternion [x, y, z, w].
 * @param out  Output direction vector [x, y, z].
 */
VRSPIN_API void vrspin_forward_vector(const double q[4], double out[3]);

/**
 * Spherical linear interpolation between two unit quaternions.
 *
 * @param q1   Start quaternion [x, y, z, w].
 * @param q2   End quaternion [x, y, z, w].
 * @param t    Interpolation factor in [0, 1].
 * @param out  Output quaternion [x, y, z, w].
 */
VRSPIN_API void vrspin_slerp(const double q1[4], const double q2[4],
                             double t, double out[4]);

/* ------------------------------------------------------------------ */
/* Attention cone — opaque handle                                       */
/* ------------------------------------------------------------------ */

/** Opaque handle to a native attention cone. */
typedef struct VRSpinCone VRSpinCone;

/**
 * Create a new attention cone.
 *
 * @param orientation  Centre quaternion [x, y, z, w].
 * @param half_angle   Half-aperture in radians.
 * @param falloff      One of ``VRSPIN_FALLOFF_*`` constants.
 * @return             Pointer to the new cone (caller owns; free with
 *                     ``vrspin_cone_destroy``).
 */
VRSPIN_API VRSpinCone *vrspin_cone_create(const double orientation[4],
                                          double half_angle,
                                          int falloff);

/**
 * Destroy an attention cone and free its memory.
 *
 * @param cone  Pointer obtained from ``vrspin_cone_create``.
 */
VRSPIN_API void vrspin_cone_destroy(VRSpinCone *cone);

/**
 * Update the cone's pointing direction.
 *
 * @param cone      Target cone.
 * @param new_quat  New centre quaternion [x, y, z, w].
 */
VRSPIN_API void vrspin_cone_update_origin(VRSpinCone *cone,
                                          const double new_quat[4]);

/**
 * Test whether a target quaternion falls inside the cone.
 *
 * @param cone    Attention cone.
 * @param target  Quaternion [x, y, z, w] to test.
 * @return        1 if inside, 0 if outside.
 */
VRSPIN_API int vrspin_cone_contains(const VRSpinCone *cone,
                                    const double target[4]);

/**
 * Compute the attenuation (attention strength) for a target quaternion.
 *
 * Returns a value in [0, 1] where 1.0 means dead-centre and 0.0 means
 * outside the cone.  The decay curve depends on the cone's falloff mode.
 *
 * @param cone    Attention cone.
 * @param target  Quaternion [x, y, z, w].
 * @return        Attenuation in [0.0, 1.0].
 */
VRSPIN_API double vrspin_cone_attenuation(const VRSpinCone *cone,
                                          const double target[4]);

/**
 * Batch membership test: which of N quaternions are inside the cone?
 *
 * @param cone     Attention cone.
 * @param quats    Flat array of N quaternions (4*N doubles, row-major).
 * @param n        Number of quaternions.
 * @param results  Output array of N ints (1 = inside, 0 = outside).
 */
VRSPIN_API void vrspin_cone_query_batch(const VRSpinCone *cone,
                                        const double *quats, int n,
                                        int *results);

/**
 * Batch attenuation: compute attention strength for N quaternions.
 *
 * @param cone     Attention cone.
 * @param quats    Flat array of N quaternions (4*N doubles, row-major).
 * @param n        Number of quaternions.
 * @param results  Output array of N doubles in [0, 1].
 */
VRSPIN_API void vrspin_cone_query_batch_attenuation(const VRSpinCone *cone,
                                                    const double *quats, int n,
                                                    double *results);

/* ------------------------------------------------------------------ */
/* Frame processing (matches WebSocket protocol but native)             */
/* ------------------------------------------------------------------ */

/**
 * Process a single attention frame for multiple entities.
 *
 * Evaluates a visual cone and an audio cone against N entity
 * orientations in a single call.  This is the recommended entry point
 * for real-time VR integration.
 *
 * @param user_quat              User head quaternion [x, y, z, w].
 * @param entity_orientations    Flat array of N entity quaternions (4*N doubles).
 * @param num_entities           Number of entities.
 * @param visual_half_angle      Visual cone half-angle in radians.
 * @param audio_half_angle       Audio cone half-angle in radians.
 * @param out_visual_strengths   Output: N visual attenuation values.
 * @param out_audio_gains        Output: N audio attenuation values.
 * @param out_visual_attended    Output: N ints (1 = visually attended).
 * @return                       Number of visually attended entities.
 */
VRSPIN_API int vrspin_process_frame(
    const double user_quat[4],
    const double *entity_orientations,
    int num_entities,
    double visual_half_angle,
    double audio_half_angle,
    double *out_visual_strengths,
    double *out_audio_gains,
    int *out_visual_attended);

/* ------------------------------------------------------------------ */
/* Version                                                              */
/* ------------------------------------------------------------------ */

/** Return the library version string. */
VRSPIN_API const char *vrspin_version(void);

#ifdef __cplusplus
}
#endif

#endif /* VRSPIN_NATIVE_H */
