/**
 * VRSpin Plaza — WebGL application entry point.
 *
 * Wires together:
 *   Input (mouse/keyboard) → Perception Engine → Behavior System → Renderer
 *
 * The scene does not respond to actions. It responds to attention.
 *
 * @module app/main
 */

import { buildPlazaScene } from "../scene/plaza.js"
import { evaluateAllPerceptions } from "../core/perception.js"
import { PlazaRenderer } from "../render/renderer.js"
import { createInputHandler } from "../render/input.js"
import { updateBehaviors } from "../render/behaviors.js"
import { createSpatialAudio } from "../audio/spatialAudio.js"

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

const canvas = document.querySelector("canvas")
if (!canvas) {
  throw new Error("No <canvas> element found in the document")
}

// Build the plaza scene (flat node list + SpinStep tree)
const { nodes } = buildPlazaScene()

// Initialize subsystems
const renderer = new PlazaRenderer(canvas)
const input = createInputHandler(canvas)
const audio = createSpatialAudio()

// Initialize audio on first user interaction (browser policy)
canvas.addEventListener("click", () => audio.init(), { once: true })
canvas.addEventListener("mousedown", () => audio.init(), { once: true })

// ---------------------------------------------------------------------------
// HUD overlay
// ---------------------------------------------------------------------------

const hud = document.getElementById("hud")

function updateHUD(headQuat, nodes) {
  if (!hud) return

  const activeNodes = nodes.filter(n => n.spinState !== "idle")
  const lines = []

  for (const node of activeNodes) {
    let info = `${node.id} [${node.spinState}]`

    // NPC greeting
    if (node.entityType === "npc" && node.npcState === "speaking") {
      info += ` — "${node.metadata.greeting}"`
    }

    // Panel page
    if (node.entityType === "panel") {
      const page = node.metadata.pages[node.metadata.currentPage || 0]
      if (page) {
        info += ` — ${page.heading}`
      }
    }

    lines.push(info)
  }

  hud.textContent = lines.length > 0
    ? lines.join("\n")
    : "Look around to discover the plaza..."
}

// ---------------------------------------------------------------------------
// Frame loop
// ---------------------------------------------------------------------------

/** Maximum frame delta (seconds) to prevent physics jumps on tab-switch. */
const MAX_DELTA_TIME = 0.1

let lastTime = performance.now()

function loop(now) {
  const deltaTime = Math.min((now - lastTime) / 1000, MAX_DELTA_TIME)
  lastTime = now

  // 1. Input — get head quaternion
  const headQuat = input.getHeadQuat()

  // 2. Perception — evaluate SpinState for all nodes
  evaluateAllPerceptions(headQuat, nodes, deltaTime)

  // 3. Behavior — update NPC slerp, panel pages, object effects
  updateBehaviors(nodes, headQuat, deltaTime)

  // 4. Audio — update spatial audio gains
  audio.update(headQuat, nodes)

  // 5. Render — draw perceived entities
  renderer.render(nodes, headQuat)

  // 6. HUD — show active entities
  updateHUD(headQuat, nodes)

  requestAnimationFrame(loop)
}

requestAnimationFrame(loop)
