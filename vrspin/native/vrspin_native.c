/**
 * @file vrspin_native.c
 * @brief Native C implementation of VRSpin attention-cone math.
 *
 * Pure C (C99) — no Python, no external dependencies beyond libm.
 * All quaternion operations match SpinStep's [x, y, z, w] convention.
 */

#include "vrspin_native.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define VRSPIN_VERSION "0.1.0"

/* ================================================================== */
/* Internal helpers                                                    */
/* ================================================================== */

static double _dot4(const double a[4], const double b[4])
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

static double _norm4(const double q[4])
{
    return sqrt(_dot4(q, q));
}

static void _normalize4(double q[4])
{
    double n = _norm4(q);
    if (n < 1e-12) return;
    q[0] /= n;
    q[1] /= n;
    q[2] /= n;
    q[3] /= n;
}

/**
 * Quaternion multiply: out = a * b  (Hamilton product).
 * Convention: [x, y, z, w].
 *
 * Retained as a building block for future operations (e.g. relative spin,
 * rotate-by-quaternion).
 */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((unused))
#endif
static void _quat_mul(const double a[4], const double b[4], double out[4])
{
    double ax = a[0], ay = a[1], az = a[2], aw = a[3];
    double bx = b[0], by = b[1], bz = b[2], bw = b[3];

    out[0] = aw * bx + ax * bw + ay * bz - az * by;
    out[1] = aw * by - ax * bz + ay * bw + az * bx;
    out[2] = aw * bz + ax * by - ay * bx + az * bw;
    out[3] = aw * bw - ax * bx - ay * by - az * bz;
}

/** Quaternion conjugate (inverse for unit quaternions). */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((unused))
#endif
static void _quat_conj(const double q[4], double out[4])
{
    out[0] = -q[0];
    out[1] = -q[1];
    out[2] = -q[2];
    out[3] =  q[3];
}

/**
 * Angular distance between two unit quaternions.
 * Returns angle in [0, pi].
 */
static double _quat_angle(const double q1[4], const double q2[4])
{
    double d = fabs(_dot4(q1, q2));
    if (d > 1.0) d = 1.0;
    return 2.0 * acos(d);
}

/**
 * Compute attenuation given angle, half_angle, and falloff mode.
 */
static double _compute_attenuation(double angle, double half_angle, int falloff)
{
    if (angle >= half_angle) return 0.0;
    if (falloff == VRSPIN_FALLOFF_NONE) return 1.0;

    double ratio = angle / half_angle;
    if (falloff == VRSPIN_FALLOFF_LINEAR)
    {
        double v = 1.0 - ratio;
        return v > 0.0 ? v : 0.0;
    }
    /* VRSPIN_FALLOFF_COSINE */
    return cos(ratio * M_PI / 2.0);
}

/* ================================================================== */
/* Cone struct (internal)                                              */
/* ================================================================== */

struct VRSpinCone
{
    double orientation[4]; /* normalised [x, y, z, w] */
    double half_angle;
    int    falloff;
};

/* ================================================================== */
/* Public API — Quaternion utilities                                    */
/* ================================================================== */

VRSPIN_API void vrspin_quat_normalize(double q[4])
{
    _normalize4(q);
}

VRSPIN_API double vrspin_quat_distance(const double q1[4], const double q2[4])
{
    double a[4], b[4];
    memcpy(a, q1, sizeof(a));
    memcpy(b, q2, sizeof(b));
    _normalize4(a);
    _normalize4(b);
    return _quat_angle(a, b);
}

VRSPIN_API void vrspin_forward_vector(const double q[4], double out[3])
{
    /*
     * Rotate [0, 0, -1] by quaternion q.
     * v' = q * v * q^-1 where v = (0, 0, -1, 0) as a pure quaternion.
     *
     * For unit quaternion q = [x, y, z, w]:
     *   R = rotation matrix columns from q
     *   forward = R * [0, 0, -1]^T = -R[:,2]
     *
     * R[:,2] = [2(xz + wy), 2(yz - wx), 1 - 2(x² + y²)]
     * so forward = [-2(xz + wy), -2(yz - wx), -(1 - 2(x² + y²))]
     */
    double x = q[0], y = q[1], z = q[2], w = q[3];

    out[0] = -2.0 * (x * z + w * y);
    out[1] = -2.0 * (y * z - w * x);
    out[2] = -(1.0 - 2.0 * (x * x + y * y));
}

VRSPIN_API void vrspin_slerp(const double q1[4], const double q2[4],
                             double t, double out[4])
{
    double a[4], b[4];
    memcpy(a, q1, sizeof(a));
    memcpy(b, q2, sizeof(b));
    _normalize4(a);
    _normalize4(b);

    double dot = _dot4(a, b);

    /* If negative dot, negate one quaternion to take the short arc. */
    if (dot < 0.0)
    {
        b[0] = -b[0]; b[1] = -b[1]; b[2] = -b[2]; b[3] = -b[3];
        dot = -dot;
    }

    if (dot > 0.9995)
    {
        /* Quaternions very close — use linear interpolation to avoid
           division by near-zero sin. */
        for (int i = 0; i < 4; i++)
            out[i] = a[i] + t * (b[i] - a[i]);
        _normalize4(out);
        return;
    }

    double theta_0 = acos(dot);        /* angle between inputs */
    double theta   = theta_0 * t;      /* angle for this step  */
    double sin_theta   = sin(theta);
    double sin_theta_0 = sin(theta_0);

    double s0 = cos(theta) - dot * sin_theta / sin_theta_0;
    double s1 = sin_theta / sin_theta_0;

    for (int i = 0; i < 4; i++)
        out[i] = s0 * a[i] + s1 * b[i];
    _normalize4(out);
}

/* ================================================================== */
/* Public API — Attention cone                                          */
/* ================================================================== */

VRSPIN_API VRSpinCone *vrspin_cone_create(const double orientation[4],
                                          double half_angle,
                                          int falloff)
{
    VRSpinCone *cone = (VRSpinCone *)malloc(sizeof(VRSpinCone));
    if (!cone) return NULL;

    memcpy(cone->orientation, orientation, 4 * sizeof(double));
    _normalize4(cone->orientation);
    cone->half_angle = half_angle;
    cone->falloff    = falloff;
    return cone;
}

VRSPIN_API void vrspin_cone_destroy(VRSpinCone *cone)
{
    free(cone);
}

VRSPIN_API void vrspin_cone_update_origin(VRSpinCone *cone,
                                          const double new_quat[4])
{
    if (!cone) return;
    memcpy(cone->orientation, new_quat, 4 * sizeof(double));
    _normalize4(cone->orientation);
}

VRSPIN_API int vrspin_cone_contains(const VRSpinCone *cone,
                                    const double target[4])
{
    if (!cone) return 0;
    double t[4];
    memcpy(t, target, sizeof(t));
    _normalize4(t);
    return _quat_angle(cone->orientation, t) < cone->half_angle ? 1 : 0;
}

VRSPIN_API double vrspin_cone_attenuation(const VRSpinCone *cone,
                                          const double target[4])
{
    if (!cone) return 0.0;
    double t[4];
    memcpy(t, target, sizeof(t));
    _normalize4(t);
    double angle = _quat_angle(cone->orientation, t);
    return _compute_attenuation(angle, cone->half_angle, cone->falloff);
}

VRSPIN_API void vrspin_cone_query_batch(const VRSpinCone *cone,
                                        const double *quats, int n,
                                        int *results)
{
    if (!cone || !quats || !results) return;
    for (int i = 0; i < n; i++)
    {
        const double *q = quats + i * 4;
        double t[4];
        memcpy(t, q, sizeof(t));
        _normalize4(t);
        results[i] = _quat_angle(cone->orientation, t) < cone->half_angle ? 1 : 0;
    }
}

VRSPIN_API void vrspin_cone_query_batch_attenuation(const VRSpinCone *cone,
                                                    const double *quats, int n,
                                                    double *results)
{
    if (!cone || !quats || !results) return;
    for (int i = 0; i < n; i++)
    {
        const double *q = quats + i * 4;
        double t[4];
        memcpy(t, q, sizeof(t));
        _normalize4(t);
        double angle = _quat_angle(cone->orientation, t);
        results[i] = _compute_attenuation(angle, cone->half_angle, cone->falloff);
    }
}

/* ================================================================== */
/* Public API — Frame processing                                        */
/* ================================================================== */

VRSPIN_API int vrspin_process_frame(
    const double user_quat[4],
    const double *entity_orientations,
    int num_entities,
    double visual_half_angle,
    double audio_half_angle,
    double *out_visual_strengths,
    double *out_audio_gains,
    int *out_visual_attended)
{
    /* Normalise user quaternion */
    double uq[4];
    memcpy(uq, user_quat, sizeof(uq));
    _normalize4(uq);

    int attended_count = 0;

    for (int i = 0; i < num_entities; i++)
    {
        const double *eq = entity_orientations + i * 4;
        double tq[4];
        memcpy(tq, eq, sizeof(tq));
        _normalize4(tq);

        double angle = _quat_angle(uq, tq);

        /* Visual (linear falloff) */
        double vis = _compute_attenuation(angle, visual_half_angle,
                                          VRSPIN_FALLOFF_LINEAR);
        out_visual_strengths[i] = vis;
        out_visual_attended[i]  = vis > 0.0 ? 1 : 0;
        if (vis > 0.0) attended_count++;

        /* Audio (cosine falloff) */
        out_audio_gains[i] = _compute_attenuation(angle, audio_half_angle,
                                                   VRSPIN_FALLOFF_COSINE);
    }

    return attended_count;
}

/* ================================================================== */
/* Version                                                              */
/* ================================================================== */

VRSPIN_API const char *vrspin_version(void)
{
    return VRSPIN_VERSION;
}
