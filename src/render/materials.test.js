import { describe, it } from "node:test"
import assert from "node:assert/strict"

import { MATERIALS, getMaterial, getStateMaterial } from "./materials.js"

// ---------------------------------------------------------------------------
// MATERIALS constants
// ---------------------------------------------------------------------------

describe("materials — MATERIALS", () => {
  it("has all expected entity types", () => {
    const keys = Object.keys(MATERIALS)
    assert.ok(keys.includes("fountain"))
    assert.ok(keys.includes("marketStand"))
    assert.ok(keys.includes("npc"))
    assert.ok(keys.includes("panel"))
    assert.ok(keys.includes("audio"))
    assert.ok(keys.includes("ground"))
  })

  it("all materials have required properties", () => {
    for (const [name, mat] of Object.entries(MATERIALS)) {
      assert.ok(Array.isArray(mat.specularColor), `${name}.specularColor`)
      assert.strictEqual(mat.specularColor.length, 3, `${name}.specularColor length`)
      assert.strictEqual(typeof mat.shininess, "number", `${name}.shininess`)
      assert.strictEqual(typeof mat.emissive, "number", `${name}.emissive`)
      assert.strictEqual(typeof mat.rimStrength, "number", `${name}.rimStrength`)
    }
  })

  it("shininess values are positive", () => {
    for (const [name, mat] of Object.entries(MATERIALS)) {
      assert.ok(mat.shininess > 0, `${name}.shininess should be > 0`)
    }
  })

  it("panel has highest emissive (self-illuminated)", () => {
    assert.ok(MATERIALS.panel.emissive > MATERIALS.fountain.emissive)
    assert.ok(MATERIALS.panel.emissive > MATERIALS.npc.emissive)
  })

  it("fountain has highest shininess among objects (wet stone)", () => {
    assert.ok(MATERIALS.fountain.shininess > MATERIALS.marketStand.shininess)
  })
})

// ---------------------------------------------------------------------------
// getMaterial
// ---------------------------------------------------------------------------

describe("materials — getMaterial", () => {
  it("returns fountain material for Fountain node", () => {
    const mat = getMaterial({ id: "Fountain", entityType: "object" })
    assert.strictEqual(mat, MATERIALS.fountain)
  })

  it("returns marketStand material for MarketStand node", () => {
    const mat = getMaterial({ id: "MarketStand", entityType: "object" })
    assert.strictEqual(mat, MATERIALS.marketStand)
  })

  it("returns npc material for any NPC", () => {
    const mat = getMaterial({ id: "Elena", entityType: "npc" })
    assert.strictEqual(mat, MATERIALS.npc)
  })

  it("returns panel material for panel entities", () => {
    const mat = getMaterial({ id: "VR Art", entityType: "panel" })
    assert.strictEqual(mat, MATERIALS.panel)
  })

  it("returns audio material for audio entities", () => {
    const mat = getMaterial({ id: "FountainAmbience", entityType: "audio" })
    assert.strictEqual(mat, MATERIALS.audio)
  })

  it("falls back to fountain for unknown object", () => {
    const mat = getMaterial({ id: "Unknown", entityType: "object" })
    assert.strictEqual(mat, MATERIALS.fountain)
  })

  it("falls back to fountain for unknown type", () => {
    const mat = getMaterial({ id: "X", entityType: "other" })
    assert.strictEqual(mat, MATERIALS.fountain)
  })
})

// ---------------------------------------------------------------------------
// getStateMaterial
// ---------------------------------------------------------------------------

describe("materials — getStateMaterial", () => {
  const base = MATERIALS.fountain

  it("perceived: zeroes specular", () => {
    const mat = getStateMaterial(base, "perceived")
    assert.deepStrictEqual(mat.specularColor, [0, 0, 0])
  })

  it("perceived: zeroes emissive", () => {
    const mat = getStateMaterial(base, "perceived")
    assert.strictEqual(mat.emissive, 0.0)
  })

  it("perceived: sets minimal rim", () => {
    const mat = getStateMaterial(base, "perceived")
    assert.strictEqual(mat.rimStrength, 0.1)
  })

  it("focused: returns base material unchanged", () => {
    const mat = getStateMaterial(base, "focused")
    assert.strictEqual(mat, base)
  })

  it("activated: boosts emissive", () => {
    const mat = getStateMaterial(base, "activated")
    assert.ok(mat.emissive > base.emissive)
  })

  it("activated: boosts rim strength", () => {
    const mat = getStateMaterial(base, "activated")
    assert.ok(mat.rimStrength > base.rimStrength)
  })

  it("activated: preserves specular color", () => {
    const mat = getStateMaterial(base, "activated")
    assert.deepStrictEqual(mat.specularColor, base.specularColor)
  })

  it("activated: clamps emissive to 1.0", () => {
    const highEmissive = { ...MATERIALS.panel, emissive: 0.9 }
    const mat = getStateMaterial(highEmissive, "activated")
    assert.ok(mat.emissive <= 1.0)
  })

  it("activated: clamps rimStrength to 1.0", () => {
    const highRim = { ...MATERIALS.panel, rimStrength: 0.95 }
    const mat = getStateMaterial(highRim, "activated")
    assert.ok(mat.rimStrength <= 1.0)
  })

  it("idle: returns base material", () => {
    const mat = getStateMaterial(base, "idle")
    assert.strictEqual(mat, base)
  })
})
