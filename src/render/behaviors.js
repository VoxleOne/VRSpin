/**
 * Entity behavior system — NPC rotation, panel pagination, object effects.
 *
 * Runs after perception evaluation. Mutates node metadata based on
 * current SpinState. No rendering logic here — only state changes.
 *
 * @module render/behaviors
 */

import { SpinState } from "../core/perception.js"
import { slerp, quaternionDistance } from "../core/spinstep.js"

/**
 * Update all node behaviors for one frame.
 *
 * @param {object[]} nodes - array of PlazaNodes
 * @param {number[]} headQuat - user head orientation [x, y, z, w]
 * @param {number} deltaTime - frame time in seconds
 * @param {Object.<string, object[]>} [npcObserverResults] - optional per-NPC observer results
 */
export function updateBehaviors(nodes, headQuat, deltaTime, npcObserverResults = null) {
  for (const node of nodes) {
    switch (node.entityType) {
      case "npc":
        updateNPC(node, headQuat, deltaTime, npcObserverResults)
        break
      case "panel":
        updatePanel(node, headQuat, deltaTime)
        break
      case "object":
        updateObject(node, headQuat, deltaTime)
        break
      // audio nodes have no visual behavior
    }
  }
}

// ---------------------------------------------------------------------------
// NPC behavior — mirrors vrspin/npc.py NPC.tick()
// ---------------------------------------------------------------------------

/**
 * NPC state machine:
 * - idle: no user in perception cone
 * - noticing: user entered cone, NPC begins SLERP toward user
 * - engaged: NPC fully facing user
 * - speaking: NPC greeting triggered
 *
 * @param {object} node - PlazaNode with entityType="npc"
 * @param {number[]} headQuat
 * @param {number} deltaTime
 * @param {Object.<string, object[]>} [npcObserverResults] - optional multi-observer results
 */
function updateNPC(node, headQuat, deltaTime, npcObserverResults = null) {
  const perceptionHalf = node.metadata.perceptionHalfAngle || (120 * Math.PI / 180)
  const slerpSpeed = node.metadata.slerpSpeed || 0.15
  const angle = quaternionDistance(headQuat, node.orientation)
  const inCone = angle < perceptionHalf

  if (inCone) {
    if (!node.npcNoticedUsers) {
      node.npcNoticedUsers = true
      node.npcState = "noticing"
      // Target: face back toward user (inverse orientation)
      // NPC turns to "look back" — use headQuat as target direction
      node.npcTargetOrientation = headQuat.slice()
    }

    if (node.npcState === "noticing" || node.npcState === "engaged") {
      // SLERP toward target
      if (node.npcTargetOrientation) {
        const t = Math.min(1.0, slerpSpeed * deltaTime * 60)
        node.metadata._currentFacing = slerp(
          node.metadata._currentFacing || node.orientation,
          node.npcTargetOrientation,
          t
        )
      }

      // Check if fully turned (< 5°)
      if (node.npcTargetOrientation && node.metadata._currentFacing) {
        const remaining = quaternionDistance(
          node.metadata._currentFacing,
          node.npcTargetOrientation
        )
        if (remaining < 5 * Math.PI / 180 && node.npcState === "noticing") {
          node.npcState = "engaged"
          if (!node.npcGreeted) {
            node.npcGreeted = true
            node.npcState = "speaking"
            node.metadata._speakingTimer = 3.0 // show greeting for 3s
          }
        }
      }
    }

    if (node.npcState === "speaking") {
      node.metadata._speakingTimer = (node.metadata._speakingTimer || 0) - deltaTime
      if (node.metadata._speakingTimer <= 0) {
        node.npcState = "engaged"
      }
    }
  } else {
    // User left cone — reset
    if (node.npcNoticedUsers) {
      node.npcNoticedUsers = false
      node.npcGreeted = false
      node.npcState = "idle"
      node.npcTargetOrientation = null
      node.metadata._currentFacing = null
    }
  }

  // NPC-as-observer: track awareness of other entities via multi-observer results
  if (npcObserverResults && npcObserverResults[node.id]) {
    const observed = npcObserverResults[node.id]
    if (!node.npcObservedEntities) node.npcObservedEntities = {}
    for (const entry of observed) {
      if (entry.state !== "idle") {
        node.npcObservedEntities[entry.node.id] = {
          state: entry.state,
          dwellTime: entry.dwellTime,
        }
      } else {
        delete node.npcObservedEntities[entry.node.id]
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Panel behavior — mirrors vrspin/entities.py KnowledgePanel
// ---------------------------------------------------------------------------

/**
 * Panel pagination:
 * - perceived: visible (fade in)
 * - focused: fully readable, show current page
 * - activated: advance to next page
 *
 * @param {object} node
 * @param {number[]} headQuat
 * @param {number} deltaTime
 */
function updatePanel(node, headQuat, deltaTime) {
  const pages = node.metadata.pages || []
  if (pages.length === 0) return

  if (node.spinState === SpinState.ACTIVATED) {
    // Advance page when activated (with cooldown to prevent rapid flipping)
    if (!node.metadata._pageCooldown || node.metadata._pageCooldown <= 0) {
      const currentPage = node.metadata.currentPage || 0
      if (currentPage < pages.length - 1) {
        node.metadata.currentPage = currentPage + 1
        node.metadata._pageCooldown = 1.0 // 1s cooldown
      }
    }
  }

  // Decrement cooldown
  if (node.metadata._pageCooldown > 0) {
    node.metadata._pageCooldown -= deltaTime
  }

  // Reset page when hidden
  if (node.spinState === SpinState.IDLE) {
    node.metadata.currentPage = 0
  }
}

// ---------------------------------------------------------------------------
// Object behavior
// ---------------------------------------------------------------------------

/**
 * Object effects:
 * - perceived: faint outline (handled in renderer)
 * - focused: highlight (handled in renderer)
 * - activated: trigger effect flag
 *
 * @param {object} node
 * @param {number[]} headQuat
 * @param {number} deltaTime
 */
function updateObject(node, headQuat, deltaTime) {
  node.metadata._activated = node.spinState === SpinState.ACTIVATED
}
