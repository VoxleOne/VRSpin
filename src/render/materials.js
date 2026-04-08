/**
 * Material definitions for plaza entity types.
 *
 * Each material specifies Blinn-Phong parameters:
 *   specularColor  — RGB highlight color
 *   shininess      — exponent (higher = tighter highlight)
 *   emissive       — self-illumination [0, 1]
 *   rimStrength    — fresnel rim light intensity [0, 1]
 *
 * Materials are looked up per-node (entity-specific overrides) and then
 * modified per-frame based on the node's SpinState.
 *
 * @module render/materials
 */

/**
 * @typedef {object} Material
 * @property {number[]} specularColor - RGB specular highlight color
 * @property {number} shininess - Specular exponent
 * @property {number} emissive - Self-illumination strength [0, 1]
 * @property {number} rimStrength - Rim/fresnel light intensity [0, 1]
 */

/** Predefined materials for each entity class. */
export const MATERIALS = Object.freeze({
  /** Wet stone — high specular, moderate rim. */
  fountain: Object.freeze({
    specularColor: [0.9, 0.9, 1.0],
    shininess: 64.0,
    emissive: 0.0,
    rimStrength: 0.3,
  }),
  /** Wood/fabric — low specular, matte. */
  marketStand: Object.freeze({
    specularColor: [0.2, 0.15, 0.1],
    shininess: 8.0,
    emissive: 0.0,
    rimStrength: 0.1,
  }),
  /** Soft organic — moderate specular, strong rim for silhouette. */
  npc: Object.freeze({
    specularColor: [0.3, 0.3, 0.3],
    shininess: 16.0,
    emissive: 0.0,
    rimStrength: 0.4,
  }),
  /** Holographic display — sharp specular, self-illuminated. */
  panel: Object.freeze({
    specularColor: [1.0, 1.0, 1.0],
    shininess: 128.0,
    emissive: 0.6,
    rimStrength: 0.5,
  }),
  /** Sound ripple — soft glow, semi-transparent feel. */
  audio: Object.freeze({
    specularColor: [0.5, 0.5, 0.5],
    shininess: 16.0,
    emissive: 0.3,
    rimStrength: 0.2,
  }),
  /** Ground — very low specular, no rim. */
  ground: Object.freeze({
    specularColor: [0.1, 0.1, 0.1],
    shininess: 4.0,
    emissive: 0.0,
    rimStrength: 0.0,
  }),
})

/**
 * Look up the base material for a node.
 * Uses node.id for entity-specific overrides, falls back to entityType.
 *
 * @param {object} node - PlazaNode
 * @returns {Material}
 */
export function getMaterial(node) {
  if (node.id === "Fountain") return MATERIALS.fountain
  if (node.id === "MarketStand") return MATERIALS.marketStand
  switch (node.entityType) {
    case "npc": return MATERIALS.npc
    case "panel": return MATERIALS.panel
    case "audio": return MATERIALS.audio
    case "object": return MATERIALS.fountain
    default: return MATERIALS.fountain
  }
}

/**
 * Apply SpinState-dependent modifications to a base material.
 *
 * - perceived: no specular, no emissive, minimal rim (soft focus)
 * - focused: unchanged (full material)
 * - activated: boosted emissive + rim (glowing highlight)
 *
 * @param {Material} base
 * @param {string} spinState
 * @returns {Material}
 */
export function getStateMaterial(base, spinState) {
  switch (spinState) {
    case "perceived":
      return {
        specularColor: [0, 0, 0],
        shininess: base.shininess,
        emissive: 0.0,
        rimStrength: 0.1,
      }
    case "activated":
      return {
        specularColor: base.specularColor,
        shininess: base.shininess,
        emissive: Math.min(1.0, base.emissive + 0.3),
        rimStrength: Math.min(1.0, base.rimStrength + 0.2),
      }
    default:
      return base
  }
}
