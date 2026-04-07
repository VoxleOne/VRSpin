/**
 * SpinStep 0.3.1a1 — JavaScript port of core quaternion math and tree traversal.
 *
 * Quaternion convention: [x, y, z, w] matching SpinStep Python.
 * Forward direction: -Z axis (identity quaternion looks toward [0, 0, -1]).
 *
 * @module core/spinstep
 */

// ---------------------------------------------------------------------------
// Quaternion math helpers
// ---------------------------------------------------------------------------

/**
 * Normalize a quaternion in-place, returning the same array.
 * @param {number[]} q - [x, y, z, w]
 * @returns {number[]}
 */
export function quatNormalize(q) {
  const len = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
  if (len < 1e-8) return [0, 0, 0, 1]
  return [q[0] / len, q[1] / len, q[2] / len, q[3] / len]
}

/**
 * Hamilton product of two quaternions in [x, y, z, w] order.
 * @param {number[]} a
 * @param {number[]} b
 * @returns {number[]}
 */
export function quatMultiply(a, b) {
  const [ax, ay, az, aw] = a
  const [bx, by, bz, bw] = b
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ]
}

/**
 * Conjugate (inverse for unit quaternions) in [x, y, z, w] order.
 * @param {number[]} q
 * @returns {number[]}
 */
export function quatConjugate(q) {
  return [-q[0], -q[1], -q[2], q[3]]
}

/**
 * Angular distance in radians between two unit quaternions.
 * Mirrors spinstep.utils.quaternion_distance.
 * @param {number[]} q1
 * @param {number[]} q2
 * @returns {number}
 */
export function quaternionDistance(q1, q2) {
  // relative rotation = q1^-1 * q2
  const inv = quatConjugate(q1)
  const rel = quatMultiply(inv, q2)
  // rotation angle = 2 * acos(|w|)
  const w = Math.min(1.0, Math.max(-1.0, Math.abs(rel[3])))
  return 2 * Math.acos(w)
}

/**
 * Check if two quaternions are within an angular threshold.
 * Mirrors spinstep.utils.is_within_angle_threshold.
 * @param {number[]} q1
 * @param {number[]} q2
 * @param {number} thresholdRad
 * @returns {boolean}
 */
export function isWithinAngle(q1, q2, thresholdRad) {
  return quaternionDistance(q1, q2) < thresholdRad
}

/**
 * Rotate a vector by a quaternion: q * [vx, vy, vz, 0] * q^-1.
 * @param {number[]} q - unit quaternion [x, y, z, w]
 * @param {number[]} v - 3D vector [x, y, z]
 * @returns {number[]}
 */
export function quatRotateVec3(q, v) {
  const vq = [v[0], v[1], v[2], 0]
  const r = quatMultiply(quatMultiply(q, vq), quatConjugate(q))
  return [r[0], r[1], r[2]]
}

/**
 * Extract forward direction (-Z) from a quaternion.
 * Mirrors spinstep.utils.forward_vector_from_quaternion.
 * @param {number[]} q - unit quaternion [x, y, z, w]
 * @returns {number[]} unit direction vector [x, y, z]
 */
export function forwardFromQuat(q) {
  return quatRotateVec3(q, [0, 0, -1])
}

/**
 * Create a quaternion from a Y-axis rotation in degrees.
 * Mirrors scipy.spatial.transform.Rotation.from_euler('y', deg, degrees=True).as_quat().
 * @param {number} deg - angle in degrees
 * @returns {number[]} [x, y, z, w]
 */
export function quatFromYDeg(deg) {
  const rad = deg * Math.PI / 180
  // Rotation around Y: [0, sin(θ/2), 0, cos(θ/2)]
  return [0, Math.sin(rad / 2), 0, Math.cos(rad / 2)]
}

/**
 * Spherical linear interpolation between two quaternions.
 * Mirrors vrspin.utils.slerp.
 * @param {number[]} q1 - start quaternion [x, y, z, w]
 * @param {number[]} q2 - end quaternion [x, y, z, w]
 * @param {number} t - interpolation parameter [0, 1]
 * @returns {number[]}
 */
export function slerp(q1, q2, t) {
  let a = quatNormalize(q1)
  let b = quatNormalize(q2)

  // Compute dot product
  let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

  // If negative dot, negate one to take shorter path
  if (dot < 0) {
    b = [-b[0], -b[1], -b[2], -b[3]]
    dot = -dot
  }

  // If nearly identical, use linear interpolation
  if (dot > 0.9995) {
    return quatNormalize([
      a[0] + t * (b[0] - a[0]),
      a[1] + t * (b[1] - a[1]),
      a[2] + t * (b[2] - a[2]),
      a[3] + t * (b[3] - a[3]),
    ])
  }

  const theta = Math.acos(dot)
  const sinTheta = Math.sin(theta)
  const wa = Math.sin((1 - t) * theta) / sinTheta
  const wb = Math.sin(t * theta) / sinTheta

  return quatNormalize([
    wa * a[0] + wb * b[0],
    wa * a[1] + wb * b[1],
    wa * a[2] + wb * b[2],
    wa * a[3] + wb * b[3],
  ])
}

// ---------------------------------------------------------------------------
// SpinStep Node
// ---------------------------------------------------------------------------

/**
 * A tree node with quaternion orientation, mirroring spinstep.Node.
 */
export class SpinNode {
  /**
   * @param {string} name
   * @param {number[]} orientation - [x, y, z, w]
   * @param {SpinNode[]} [children]
   */
  constructor(name, orientation, children = []) {
    this.name = name
    this.orientation = quatNormalize(orientation)
    this.children = children
  }
}

// ---------------------------------------------------------------------------
// QuaternionDepthIterator
// ---------------------------------------------------------------------------

/**
 * When no explicit angle_threshold is given, the threshold is set to
 * this fraction of the rotation step's angle. This mirrors the Python
 * QuaternionDepthIterator which uses 30 % of the step angle as the
 * default threshold so that only children closely aligned with the
 * rotated state are visited.
 */
const DEFAULT_DYNAMIC_THRESHOLD_FACTOR = 0.3

/**
 * Depth-first tree iterator driven by a quaternion rotation step.
 * Mirrors spinstep.QuaternionDepthIterator.
 *
 * @param {SpinNode} root
 * @param {number[]} rotationStepQuat - [x, y, z, w]
 * @param {number|null} [angleThreshold] - radians; null for auto
 * @returns {SpinNode[]} visited nodes
 */
export function quaternionDepthIterate(root, rotationStepQuat, angleThreshold = null) {
  const stepQuat = quatNormalize(rotationStepQuat)

  // Compute step angle magnitude
  const stepW = Math.min(1.0, Math.max(-1.0, Math.abs(stepQuat[3])))
  const stepAngle = 2 * Math.acos(stepW)

  let threshold
  if (angleThreshold !== null) {
    threshold = angleThreshold
  } else if (stepAngle < 1e-7) {
    threshold = Math.PI / 180 // 1 degree
  } else {
    threshold = stepAngle * DEFAULT_DYNAMIC_THRESHOLD_FACTOR
  }

  const visited = []
  // Stack: [node, currentStateQuat]
  const stack = [[root, root.orientation]]

  while (stack.length > 0) {
    const [node, state] = stack.pop()

    // Apply rotation step to current state
    const rotatedState = quatMultiply(state, stepQuat)

    for (const child of node.children) {
      const childNorm = Math.sqrt(
        child.orientation[0] ** 2 + child.orientation[1] ** 2 +
        child.orientation[2] ** 2 + child.orientation[3] ** 2
      )
      if (childNorm < 1e-8) continue

      // Angle between rotatedState and child orientation
      const angle = quaternionDistance(rotatedState, child.orientation)
      if (angle < threshold) {
        stack.push([child, child.orientation])
      }
    }

    visited.push(node)
  }

  return visited
}
