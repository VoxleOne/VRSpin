import { describe, it } from "node:test"
import assert from "node:assert/strict"

import { updateBehaviors } from "./behaviors.js"
import { SpinState } from "../core/perception.js"
import { quatFromYDeg, quaternionDistance } from "../core/spinstep.js"

function makeNPCNode() {
  return {
    id: "TestNPC",
    orientation: quatFromYDeg(0),
    entityType: "npc",
    spinState: SpinState.PERCEIVED,
    dwellTime: 0.5,
    npcState: "idle",
    npcNoticedUsers: false,
    npcGreeted: false,
    npcTargetOrientation: null,
    metadata: {
      greeting: "Hello!",
      perceptionHalfAngle: 120 * Math.PI / 180,
      slerpSpeed: 0.5,
      color: [0.8, 0.3, 0.5],
    },
  }
}

function makePanelNode() {
  return {
    id: "TestPanel",
    orientation: quatFromYDeg(70),
    entityType: "panel",
    spinState: SpinState.IDLE,
    dwellTime: 0,
    metadata: {
      pages: [
        { heading: "Page 1", body: "Content 1" },
        { heading: "Page 2", body: "Content 2" },
      ],
      currentPage: 0,
      color: [0.2, 0.5, 0.9],
    },
  }
}

function makeObjectNode() {
  return {
    id: "TestObject",
    orientation: quatFromYDeg(0),
    entityType: "object",
    spinState: SpinState.IDLE,
    dwellTime: 0,
    metadata: {
      description: "Test object",
      color: [0.9, 0.6, 0.2],
    },
  }
}

describe("behaviors — NPC", () => {
  it("NPC notices user in perception cone", () => {
    const node = makeNPCNode()
    node.metadata.slerpSpeed = 0.01 // slow slerp to stay in noticing
    const headQuat = quatFromYDeg(50) // far enough that slerp takes multiple frames
    updateBehaviors([node], headQuat, 1 / 60)
    assert.strictEqual(node.npcNoticedUsers, true)
    assert.strictEqual(node.npcState, "noticing")
  })

  it("NPC does not notice user outside cone", () => {
    const node = makeNPCNode()
    const headQuat = quatFromYDeg(170) // far away
    updateBehaviors([node], headQuat, 1 / 60)
    assert.strictEqual(node.npcNoticedUsers, false)
    assert.strictEqual(node.npcState, "idle")
  })

  it("NPC begins SLERP toward user when noticing", () => {
    const node = makeNPCNode()
    const headQuat = quatFromYDeg(10)
    updateBehaviors([node], headQuat, 1 / 60)
    assert.ok(node.npcTargetOrientation !== null)
    assert.ok(node.metadata._currentFacing !== null)
  })

  it("NPC resets when user leaves cone", () => {
    const node = makeNPCNode()
    const headQuat = quatFromYDeg(5)
    // Enter cone
    updateBehaviors([node], headQuat, 1 / 60)
    assert.strictEqual(node.npcNoticedUsers, true)
    // Leave cone
    updateBehaviors([node], quatFromYDeg(170), 1 / 60)
    assert.strictEqual(node.npcNoticedUsers, false)
    assert.strictEqual(node.npcState, "idle")
  })
})

describe("behaviors — Panel", () => {
  it("resets page when panel goes idle", () => {
    const node = makePanelNode()
    node.metadata.currentPage = 1
    node.spinState = SpinState.IDLE
    updateBehaviors([node], quatFromYDeg(0), 1 / 60)
    assert.strictEqual(node.metadata.currentPage, 0)
  })

  it("advances page when activated", () => {
    const node = makePanelNode()
    node.spinState = SpinState.ACTIVATED
    updateBehaviors([node], quatFromYDeg(70), 1 / 60)
    assert.strictEqual(node.metadata.currentPage, 1)
  })

  it("does not advance past last page", () => {
    const node = makePanelNode()
    node.metadata.currentPage = 1 // already on last page
    node.spinState = SpinState.ACTIVATED
    updateBehaviors([node], quatFromYDeg(70), 1 / 60)
    assert.strictEqual(node.metadata.currentPage, 1)
  })

  it("cooldown prevents rapid page flipping", () => {
    const node = makePanelNode()
    node.spinState = SpinState.ACTIVATED
    // First advance
    updateBehaviors([node], quatFromYDeg(70), 1 / 60)
    assert.strictEqual(node.metadata.currentPage, 1)
    // Try again immediately — should be blocked by cooldown
    node.metadata.currentPage = 0 // reset to test
    updateBehaviors([node], quatFromYDeg(70), 1 / 60)
    assert.strictEqual(node.metadata.currentPage, 0) // blocked
  })
})

describe("behaviors — Object", () => {
  it("sets activated flag when state is activated", () => {
    const node = makeObjectNode()
    node.spinState = SpinState.ACTIVATED
    updateBehaviors([node], quatFromYDeg(0), 1 / 60)
    assert.strictEqual(node.metadata._activated, true)
  })

  it("clears activated flag when not activated", () => {
    const node = makeObjectNode()
    node.spinState = SpinState.FOCUSED
    updateBehaviors([node], quatFromYDeg(0), 1 / 60)
    assert.strictEqual(node.metadata._activated, false)
  })
})

describe("behaviors — updateBehaviors", () => {
  it("handles mixed node types without errors", () => {
    const nodes = [makeNPCNode(), makePanelNode(), makeObjectNode()]
    // Should not throw
    updateBehaviors(nodes, quatFromYDeg(0), 1 / 60)
  })

  it("handles audio nodes gracefully (no-op)", () => {
    const audioNode = {
      id: "TestAudio",
      orientation: quatFromYDeg(0),
      entityType: "audio",
      spinState: SpinState.PERCEIVED,
      dwellTime: 0,
      metadata: { baseVolume: 0.7 },
    }
    // Should not throw
    updateBehaviors([audioNode], quatFromYDeg(0), 1 / 60)
  })
})
