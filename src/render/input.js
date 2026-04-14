/**
 * Mouse/keyboard input handler — desktop fallback for WebXR.
 *
 * Produces a head quaternion [x, y, z, w] each frame from mouse drag
 * (yaw/pitch) or arrow keys. The quaternion interface is identical to
 * what WebXR would provide, making the perception engine input-agnostic.
 *
 * @module render/input
 */

import { quatNormalize, quatMultiply, quatFromYDeg, quatFromXDeg } from "../core/spinstep.js"

/**
 * Create an input handler attached to a canvas.
 *
 * @param {HTMLCanvasElement} canvas
 * @returns {{ getHeadQuat: () => number[], dispose: () => void }}
 */
export function createInputHandler(canvas) {
  let yawDeg = 0    // Y-axis rotation (left/right)
  let pitchDeg = 0  // X-axis rotation (up/down)

  let isDragging = false
  let lastX = 0
  let lastY = 0

  /** Mouse drag sensitivity (degrees per pixel). */
  const SENSITIVITY = 0.3
  /** Keyboard rotation speed (degrees per frame). */
  const KEY_SPEED = 2.0

  // Track pressed keys
  const keys = new Set()

  // --- Mouse handlers ---
  function onMouseDown(e) {
    isDragging = true
    lastX = e.clientX
    lastY = e.clientY
  }

  function onMouseUp() {
    isDragging = false
  }

  function onMouseMove(e) {
    if (!isDragging) return
    const dx = e.clientX - lastX
    const dy = e.clientY - lastY
    lastX = e.clientX
    lastY = e.clientY

    yawDeg -= dx * SENSITIVITY
    pitchDeg -= dy * SENSITIVITY
    pitchDeg = Math.max(-89, Math.min(89, pitchDeg))
  }

  // --- Key handlers ---
  function onKeyDown(e) {
    keys.add(e.key)
  }

  function onKeyUp(e) {
    keys.delete(e.key)
  }

  // Bind events
  canvas.addEventListener("mousedown", onMouseDown)
  window.addEventListener("mouseup", onMouseUp)
  window.addEventListener("mousemove", onMouseMove)
  window.addEventListener("keydown", onKeyDown)
  window.addEventListener("keyup", onKeyUp)

  return {
    /**
     * Get current head quaternion. Call once per frame.
     * Also processes held keys.
     * @returns {number[]} [x, y, z, w]
     */
    getHeadQuat() {
      // Process held keys
      if (keys.has("ArrowLeft") || keys.has("a")) yawDeg += KEY_SPEED
      if (keys.has("ArrowRight") || keys.has("d")) yawDeg -= KEY_SPEED
      if (keys.has("ArrowUp") || keys.has("w")) {
        pitchDeg += KEY_SPEED
        pitchDeg = Math.min(89, pitchDeg)
      }
      if (keys.has("ArrowDown") || keys.has("s")) {
        pitchDeg -= KEY_SPEED
        pitchDeg = Math.max(-89, pitchDeg)
      }

      // Build quaternion: yaw (Y) * pitch (X) using spinstep helpers
      const qYaw = quatFromYDeg(yawDeg)
      const qPitch = quatFromXDeg(pitchDeg)

      return quatNormalize(quatMultiply(qYaw, qPitch))
    },

    /**
     * Clean up event listeners.
     */
    dispose() {
      canvas.removeEventListener("mousedown", onMouseDown)
      window.removeEventListener("mouseup", onMouseUp)
      window.removeEventListener("mousemove", onMouseMove)
      window.removeEventListener("keydown", onKeyDown)
      window.removeEventListener("keyup", onKeyUp)
    },
  }
}
