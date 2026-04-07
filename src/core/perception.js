/**
 * Perception engine — attention cone evaluation and state machine.
 *
 * Implements a 4-state perception model (idle → perceived → focused → activated)
 * driven by gaze direction + dwell time. Mirrors VRSpin's AttentionCone
 * behavior with linear falloff attenuation.
 *
 * @module core/perception
 */

import { quaternionDistance } from "./spinstep.js"

// ---------------------------------------------------------------------------
// SpinState enum
// ---------------------------------------------------------------------------

/** @enum {string} */
export const SpinState = Object.freeze({
  IDLE: "idle",
  PERCEIVED: "perceived",
  FOCUSED: "focused",
  ACTIVATED: "activated",
})

// ---------------------------------------------------------------------------
// Dwell time thresholds (seconds)
// ---------------------------------------------------------------------------

/** Time in cone before transitioning idle → perceived. */
export const DWELL_PERCEIVED = 0.0
/** Time in cone before transitioning perceived → focused. */
export const DWELL_FOCUSED = 0.5
/** Time in cone before transitioning focused → activated. */
export const DWELL_ACTIVATED = 2.0

// ---------------------------------------------------------------------------
// Attention cone defaults (radians) — matching VRUser Python class
// ---------------------------------------------------------------------------

/** Visual cone half-angle (60°). */
export const VISUAL_HALF_ANGLE = Math.PI / 3
/** Audio cone half-angle (120°). */
export const AUDIO_HALF_ANGLE = (2 * Math.PI) / 3
/** Haptic cone half-angle (30°). */
export const HAPTIC_HALF_ANGLE = Math.PI / 6

// ---------------------------------------------------------------------------
// Attenuation
// ---------------------------------------------------------------------------

/**
 * Linear attenuation within a cone.
 * Returns 1.0 at center, 0.0 at edge, 0.0 outside.
 *
 * @param {number} angle - angular distance in radians
 * @param {number} halfAngle - cone half-angle in radians
 * @returns {number} attenuation in [0, 1]
 */
export function linearAttenuation(angle, halfAngle) {
  if (angle >= halfAngle) return 0.0
  return 1.0 - angle / halfAngle
}

/**
 * Cosine attenuation within a cone.
 * Returns 1.0 at center, 0.0 at edge, 0.0 outside.
 *
 * @param {number} angle - angular distance in radians
 * @param {number} halfAngle - cone half-angle in radians
 * @returns {number} attenuation in [0, 1]
 */
export function cosineAttenuation(angle, halfAngle) {
  if (angle >= halfAngle) return 0.0
  return Math.cos((angle / halfAngle) * Math.PI / 2)
}

// ---------------------------------------------------------------------------
// Perception evaluation
// ---------------------------------------------------------------------------

/**
 * Evaluate perception state for a single node given a head quaternion.
 *
 * Mutates node.dwellTime and node.spinState in place.
 *
 * @param {number[]} headQuat - user head orientation [x, y, z, w]
 * @param {object} node - PlazaNode with .orientation, .dwellTime, .spinState
 * @param {number} deltaTime - frame time in seconds
 * @param {number} [halfAngle=VISUAL_HALF_ANGLE] - cone half-angle in radians
 */
export function evaluatePerception(headQuat, node, deltaTime, halfAngle = VISUAL_HALF_ANGLE) {
  const angle = quaternionDistance(headQuat, node.orientation)

  if (angle < halfAngle) {
    // Inside cone — accumulate dwell time
    node.dwellTime += deltaTime

    if (node.dwellTime >= DWELL_ACTIVATED) {
      node.spinState = SpinState.ACTIVATED
    } else if (node.dwellTime >= DWELL_FOCUSED) {
      node.spinState = SpinState.FOCUSED
    } else {
      node.spinState = SpinState.PERCEIVED
    }
  } else {
    // Outside cone — reset
    node.dwellTime = 0
    node.spinState = SpinState.IDLE
  }
}

/**
 * Evaluate perception for all nodes in a scene.
 *
 * @param {number[]} headQuat - user head orientation [x, y, z, w]
 * @param {object[]} nodes - array of PlazaNodes
 * @param {number} deltaTime - frame time in seconds
 * @param {number} [halfAngle=VISUAL_HALF_ANGLE]
 */
export function evaluateAllPerceptions(headQuat, nodes, deltaTime, halfAngle = VISUAL_HALF_ANGLE) {
  for (const node of nodes) {
    evaluatePerception(headQuat, node, deltaTime, halfAngle)
  }
}

/**
 * Compute audio gain for a node based on angular distance.
 * Uses cosine falloff within audio cone, attenuated to 15% outside
 * (matching plaza.py behavior).
 *
 * @param {number[]} headQuat
 * @param {number[]} nodeOrientation
 * @param {number} baseVolume
 * @returns {number} gain in [0, 1]
 */
export function computeAudioGain(headQuat, nodeOrientation, baseVolume) {
  const angle = quaternionDistance(headQuat, nodeOrientation)
  if (angle < AUDIO_HALF_ANGLE) {
    return baseVolume * cosineAttenuation(angle, AUDIO_HALF_ANGLE)
  }
  return baseVolume * 0.15
}
