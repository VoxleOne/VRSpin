/**
 * Spatial audio manager — Web Audio API integration for orientation-driven sound.
 *
 * Each audio source node gets a GainNode whose gain is computed from
 * angular alignment between head orientation and source orientation.
 * Uses cosine falloff inside the audio cone (120° half-angle) and
 * attenuates to 15% outside (matching plaza.py behavior).
 *
 * @module audio/spatialAudio
 */

import { computeAudioGain } from "../core/perception.js"

/**
 * Create a spatial audio manager.
 *
 * @returns {{ init: () => void, update: (headQuat: number[], nodes: object[]) => void, dispose: () => void }}
 */
export function createSpatialAudio() {
  /** @type {AudioContext|null} */
  let ctx = null
  /** @type {Map<string, { gainNode: GainNode, oscillator: OscillatorNode }>} */
  const sources = new Map()
  let initialized = false

  return {
    /**
     * Initialize the AudioContext. Must be called from a user gesture.
     */
    init() {
      if (initialized) return
      try {
        ctx = new AudioContext()
        initialized = true
      } catch {
        // Web Audio not available — silent fallback
        initialized = false
      }
    },

    /**
     * Update audio gains for all audio-type nodes.
     * @param {number[]} headQuat - [x, y, z, w]
     * @param {object[]} nodes - all PlazaNodes
     */
    update(headQuat, nodes) {
      if (!ctx || ctx.state === "suspended") {
        if (ctx) ctx.resume().catch(() => {})
        return
      }

      for (const node of nodes) {
        if (node.entityType !== "audio") continue

        const baseVolume = node.metadata.baseVolume || 0.7
        const gain = computeAudioGain(headQuat, node.orientation, baseVolume)

        if (!sources.has(node.id)) {
          // Create a procedural audio source (oscillator → gain → output)
          const gainNode = ctx.createGain()
          gainNode.gain.value = gain
          gainNode.connect(ctx.destination)

          const oscillator = ctx.createOscillator()
          // Use different tones for different sources
          if (node.id.toLowerCase().includes("fountain")) {
            // Pink noise approximation — low rumble
            oscillator.type = "sine"
            oscillator.frequency.value = 120
          } else {
            // Market music — brighter tone
            oscillator.type = "triangle"
            oscillator.frequency.value = 220
          }
          oscillator.connect(gainNode)
          oscillator.start()

          sources.set(node.id, { gainNode, oscillator })
        } else {
          // Smooth gain transition
          const { gainNode } = sources.get(node.id)
          gainNode.gain.linearRampToValueAtTime(gain, ctx.currentTime + 0.05)
        }
      }
    },

    /**
     * Clean up all audio resources.
     */
    dispose() {
      for (const [, { oscillator, gainNode }] of sources) {
        oscillator.stop()
        oscillator.disconnect()
        gainNode.disconnect()
      }
      sources.clear()
      if (ctx) {
        ctx.close().catch(() => {})
        ctx = null
      }
      initialized = false
    },
  }
}
