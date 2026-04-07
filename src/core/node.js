/**
 * PlazaNode — combines SpinStep orientation with perception state and render info.
 *
 * Each node represents a scene entity positioned on S³ (orientation space).
 * No Cartesian position is used as source of truth — orientation defines
 * spatial relationship, and position is derived for rendering.
 *
 * @module core/node
 */

import { SpinState } from "./perception.js"
import { forwardFromQuat } from "./spinstep.js"

/**
 * @typedef {'npc'|'object'|'audio'|'panel'} EntityType
 */

/**
 * Create a PlazaNode — the runtime representation of a scene entity.
 *
 * @param {object} opts
 * @param {string} opts.id
 * @param {number[]} opts.orientation - [x, y, z, w] quaternion
 * @param {EntityType} opts.entityType
 * @param {Record<string, any>} [opts.metadata={}]
 * @returns {object} PlazaNode
 */
export function createPlazaNode({ id, orientation, entityType, metadata = {} }) {
  return {
    id,
    orientation,      // S³ position (source of truth)
    entityType,       // 'npc' | 'object' | 'audio' | 'panel'
    spinState: SpinState.IDLE,
    dwellTime: 0,     // accumulated gaze seconds
    metadata,         // entity-specific data

    // NPC-specific state
    npcState: entityType === "npc" ? "idle" : null,
    npcTargetOrientation: null,
    npcNoticedUsers: false,
    npcGreeted: false,
  }
}

/**
 * Compute world-space position for a node from its orientation.
 * Projects orientation onto a sphere of given radius around the origin.
 *
 * @param {number[]} orientation - [x, y, z, w]
 * @param {number} radius - distance from camera
 * @returns {number[]} [x, y, z] world position
 */
export function orientationToPosition(orientation, radius) {
  const fwd = forwardFromQuat(orientation)
  return [fwd[0] * radius, fwd[1] * radius, fwd[2] * radius]
}
