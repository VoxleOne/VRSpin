import { describe, it } from "node:test"
import assert from "node:assert/strict"

import {
  SpinState,
  DWELL_PERCEIVED,
  DWELL_FOCUSED,
  DWELL_ACTIVATED,
  linearAttenuation,
  cosineAttenuation,
  evaluatePerception,
  evaluateAllPerceptions,
  computeAudioGain,
  VISUAL_HALF_ANGLE,
  AUDIO_HALF_ANGLE,
} from "./perception.js"

import { quatFromYDeg } from "./spinstep.js"

const EPS = 1e-6

function makeNode(orientationDeg = 0) {
  return {
    orientation: quatFromYDeg(orientationDeg),
    dwellTime: 0,
    spinState: SpinState.IDLE,
  }
}

describe("perception — attenuation", () => {
  it("linearAttenuation returns 1 at center", () => {
    assert.ok(Math.abs(linearAttenuation(0, 1.0) - 1.0) < EPS)
  })

  it("linearAttenuation returns 0 at edge", () => {
    assert.ok(Math.abs(linearAttenuation(1.0, 1.0)) < EPS)
  })

  it("linearAttenuation returns 0 outside cone", () => {
    assert.ok(Math.abs(linearAttenuation(1.5, 1.0)) < EPS)
  })

  it("linearAttenuation returns 0.5 at half angle", () => {
    assert.ok(Math.abs(linearAttenuation(0.5, 1.0) - 0.5) < EPS)
  })

  it("cosineAttenuation returns 1 at center", () => {
    assert.ok(Math.abs(cosineAttenuation(0, 1.0) - 1.0) < EPS)
  })

  it("cosineAttenuation returns 0 outside cone", () => {
    assert.ok(Math.abs(cosineAttenuation(1.5, 1.0)) < EPS)
  })
})

describe("perception — evaluatePerception", () => {
  it("node starts idle", () => {
    const node = makeNode(0)
    assert.strictEqual(node.spinState, SpinState.IDLE)
  })

  it("node inside cone transitions to perceived immediately", () => {
    const node = makeNode(10) // 10° away
    const head = quatFromYDeg(10) // looking at it
    evaluatePerception(head, node, 0.016)
    assert.strictEqual(node.spinState, SpinState.PERCEIVED)
  })

  it("node outside cone stays idle", () => {
    const node = makeNode(90) // 90° away
    const head = quatFromYDeg(0) // looking straight
    evaluatePerception(head, node, 0.016)
    assert.strictEqual(node.spinState, SpinState.IDLE)
  })

  it("dwell accumulates and transitions to focused", () => {
    const node = makeNode(0)
    const head = quatFromYDeg(0)
    // Simulate 0.6s of looking (>0.5s threshold)
    for (let i = 0; i < 38; i++) {
      evaluatePerception(head, node, 1 / 60)
    }
    assert.strictEqual(node.spinState, SpinState.FOCUSED)
    assert.ok(node.dwellTime >= DWELL_FOCUSED)
  })

  it("dwell accumulates and transitions to activated", () => {
    const node = makeNode(0)
    const head = quatFromYDeg(0)
    // Simulate 2.1s of looking (>2.0s threshold)
    for (let i = 0; i < 126; i++) {
      evaluatePerception(head, node, 1 / 60)
    }
    assert.strictEqual(node.spinState, SpinState.ACTIVATED)
    assert.ok(node.dwellTime >= DWELL_ACTIVATED)
  })

  it("looking away resets dwell and state", () => {
    const node = makeNode(0)
    const head = quatFromYDeg(0)
    // Build up dwell
    for (let i = 0; i < 60; i++) {
      evaluatePerception(head, node, 1 / 60)
    }
    assert.strictEqual(node.spinState, SpinState.FOCUSED)
    // Look away
    const headAway = quatFromYDeg(90)
    evaluatePerception(headAway, node, 1 / 60)
    assert.strictEqual(node.spinState, SpinState.IDLE)
    assert.ok(Math.abs(node.dwellTime) < EPS)
  })
})

describe("perception — evaluateAllPerceptions", () => {
  it("evaluates multiple nodes", () => {
    const nodes = [makeNode(0), makeNode(10), makeNode(90)]
    const head = quatFromYDeg(5)
    evaluateAllPerceptions(head, nodes, 0.016)
    assert.strictEqual(nodes[0].spinState, SpinState.PERCEIVED)
    assert.strictEqual(nodes[1].spinState, SpinState.PERCEIVED)
    assert.strictEqual(nodes[2].spinState, SpinState.IDLE)
  })
})

describe("perception — computeAudioGain", () => {
  it("returns full volume when looking directly at source", () => {
    const gain = computeAudioGain(quatFromYDeg(0), quatFromYDeg(0), 0.8)
    assert.ok(Math.abs(gain - 0.8) < 0.01)
  })

  it("returns attenuated volume when outside audio cone", () => {
    // 130° away — outside 120° audio cone
    const gain = computeAudioGain(quatFromYDeg(0), quatFromYDeg(130), 0.8)
    assert.ok(Math.abs(gain - 0.8 * 0.15) < 0.01)
  })

  it("returns partial volume inside audio cone edge", () => {
    // 60° away — inside 120° audio cone, should have reduced gain
    const gain = computeAudioGain(quatFromYDeg(0), quatFromYDeg(60), 0.8)
    assert.ok(gain > 0.15 * 0.8)
    assert.ok(gain < 0.8)
  })
})
