import { describe, it } from "node:test"
import assert from "node:assert/strict"

import { buildPlazaScene } from "./plaza.js"
import { SpinState } from "../core/perception.js"
import { quaternionDistance, quatFromYDeg } from "../core/spinstep.js"

const EPS = 0.01

describe("plaza — buildPlazaScene", () => {
  it("returns 8 nodes", () => {
    const { nodes } = buildPlazaScene()
    assert.strictEqual(nodes.length, 8)
  })

  it("returns a SpinStep tree root named 'plaza'", () => {
    const { root } = buildPlazaScene()
    assert.strictEqual(root.name, "plaza")
    assert.strictEqual(root.children.length, 4)
  })

  it("all nodes start in idle state", () => {
    const { nodes } = buildPlazaScene()
    for (const node of nodes) {
      assert.strictEqual(node.spinState, SpinState.IDLE)
      assert.strictEqual(node.dwellTime, 0)
    }
  })

  it("has correct entity types", () => {
    const { nodes } = buildPlazaScene()
    const byId = Object.fromEntries(nodes.map(n => [n.id, n]))
    assert.strictEqual(byId["Fountain"].entityType, "object")
    assert.strictEqual(byId["MarketStand"].entityType, "object")
    assert.strictEqual(byId["Elena"].entityType, "npc")
    assert.strictEqual(byId["Kai"].entityType, "npc")
    assert.strictEqual(byId["FountainAmbience"].entityType, "audio")
    assert.strictEqual(byId["MarketMusic"].entityType, "audio")
    assert.strictEqual(byId["VR Art"].entityType, "panel")
    assert.strictEqual(byId["Digital Sculpture"].entityType, "panel")
  })

  it("Fountain orientation matches north (identity)", () => {
    const { nodes } = buildPlazaScene()
    const fountain = nodes.find(n => n.id === "Fountain")
    assert.ok(quaternionDistance(fountain.orientation, quatFromYDeg(0)) < EPS)
  })

  it("MarketStand orientation matches east (-70°)", () => {
    const { nodes } = buildPlazaScene()
    const ms = nodes.find(n => n.id === "MarketStand")
    assert.ok(quaternionDistance(ms.orientation, quatFromYDeg(-70)) < EPS)
  })

  it("VR Art panel orientation matches NW (70°)", () => {
    const { nodes } = buildPlazaScene()
    const panel = nodes.find(n => n.id === "VR Art")
    assert.ok(quaternionDistance(panel.orientation, quatFromYDeg(70)) < EPS)
  })

  it("Digital Sculpture orientation matches west (85°)", () => {
    const { nodes } = buildPlazaScene()
    const panel = nodes.find(n => n.id === "Digital Sculpture")
    assert.ok(quaternionDistance(panel.orientation, quatFromYDeg(85)) < EPS)
  })

  it("NPC nodes have greeting metadata", () => {
    const { nodes } = buildPlazaScene()
    const elena = nodes.find(n => n.id === "Elena")
    assert.ok(elena.metadata.greeting.length > 0)
    const kai = nodes.find(n => n.id === "Kai")
    assert.ok(kai.metadata.greeting.length > 0)
  })

  it("panel nodes have pages metadata", () => {
    const { nodes } = buildPlazaScene()
    const vrArt = nodes.find(n => n.id === "VR Art")
    assert.strictEqual(vrArt.metadata.pages.length, 2)
    const ds = nodes.find(n => n.id === "Digital Sculpture")
    assert.strictEqual(ds.metadata.pages.length, 1)
  })

  it("audio nodes have baseVolume metadata", () => {
    const { nodes } = buildPlazaScene()
    const fa = nodes.find(n => n.id === "FountainAmbience")
    assert.strictEqual(fa.metadata.baseVolume, 0.7)
    const mm = nodes.find(n => n.id === "MarketMusic")
    assert.strictEqual(mm.metadata.baseVolume, 0.9)
  })

  it("tree zones have correct names", () => {
    const { root } = buildPlazaScene()
    const names = root.children.map(c => c.name)
    assert.ok(names.includes("north_zone"))
    assert.ok(names.includes("northwest_zone"))
    assert.ok(names.includes("west_zone"))
    assert.ok(names.includes("east_zone"))
  })
})
