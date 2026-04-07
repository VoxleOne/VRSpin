import { describe, it } from "node:test"
import assert from "node:assert/strict"

import { createPlazaNode, orientationToPosition } from "./node.js"
import { SpinState } from "./perception.js"
import { quatFromYDeg } from "./spinstep.js"

const EPS = 1e-5

describe("node — createPlazaNode", () => {
  it("creates a node with correct defaults", () => {
    const node = createPlazaNode({
      id: "test",
      orientation: [0, 0, 0, 1],
      entityType: "object",
    })
    assert.strictEqual(node.id, "test")
    assert.strictEqual(node.entityType, "object")
    assert.strictEqual(node.spinState, SpinState.IDLE)
    assert.strictEqual(node.dwellTime, 0)
    assert.deepStrictEqual(node.metadata, {})
  })

  it("NPC nodes have npc-specific state", () => {
    const node = createPlazaNode({
      id: "npc1",
      orientation: [0, 0, 0, 1],
      entityType: "npc",
    })
    assert.strictEqual(node.npcState, "idle")
    assert.strictEqual(node.npcNoticedUsers, false)
    assert.strictEqual(node.npcGreeted, false)
  })

  it("non-NPC nodes have null npcState", () => {
    const node = createPlazaNode({
      id: "obj",
      orientation: [0, 0, 0, 1],
      entityType: "object",
    })
    assert.strictEqual(node.npcState, null)
  })

  it("stores metadata", () => {
    const node = createPlazaNode({
      id: "test",
      orientation: [0, 0, 0, 1],
      entityType: "object",
      metadata: { color: [1, 0, 0] },
    })
    assert.deepStrictEqual(node.metadata.color, [1, 0, 0])
  })
})

describe("node — orientationToPosition", () => {
  it("identity quaternion maps to [0, 0, -radius] (forward = -Z)", () => {
    const pos = orientationToPosition([0, 0, 0, 1], 5)
    assert.ok(Math.abs(pos[0]) < EPS)
    assert.ok(Math.abs(pos[1]) < EPS)
    assert.ok(Math.abs(pos[2] - (-5)) < EPS)
  })

  it("90° Y rotation maps to [-radius, 0, 0]", () => {
    const pos = orientationToPosition(quatFromYDeg(90), 5)
    assert.ok(Math.abs(pos[0] - (-5)) < EPS)
    assert.ok(Math.abs(pos[1]) < EPS)
    assert.ok(Math.abs(pos[2]) < EPS)
  })

  it("position magnitude equals radius", () => {
    const pos = orientationToPosition(quatFromYDeg(45), 7)
    const mag = Math.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)
    assert.ok(Math.abs(mag - 7) < EPS)
  })
})
