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
  evaluateMultiObserverPerceptions,
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

describe("perception — evaluateMultiObserverPerceptions", () => {
  it("test_two_observers_independent_dwell", () => {
    const observers = [
      { id: "Alice", quat: quatFromYDeg(0) },
      { id: "Bob", quat: quatFromYDeg(90) },
    ]
    const nodes = [
      { ...makeNode(0), id: "N1" },
      { ...makeNode(45), id: "N2" },
      { ...makeNode(90), id: "N3" },
    ]

    const results = evaluateMultiObserverPerceptions(observers, nodes, 0.016)

    // Alice looks at 0° — should see N1 (0°) and N2 (45°), not N3 (90°)
    assert.ok(results["Alice"])
    assert.strictEqual(results["Alice"].length, 3)
    const aliceN1 = results["Alice"].find(r => r.node.id === "N1")
    const aliceN3 = results["Alice"].find(r => r.node.id === "N3")
    assert.notStrictEqual(aliceN1.state, SpinState.IDLE)
    assert.strictEqual(aliceN3.state, SpinState.IDLE)

    // Bob looks at 90° — should see N3 (90°) and N2 (45°), not N1 (0°)
    assert.ok(results["Bob"])
    assert.strictEqual(results["Bob"].length, 3)
    const bobN3 = results["Bob"].find(r => r.node.id === "N3")
    const bobN1 = results["Bob"].find(r => r.node.id === "N1")
    assert.notStrictEqual(bobN3.state, SpinState.IDLE)
    assert.strictEqual(bobN1.state, SpinState.IDLE)

    // Dwell times are independent per observer
    assert.ok(aliceN1.dwellTime > 0)
    assert.strictEqual(bobN1.dwellTime, 0)
  })

  it("test_self_observation_skipped", () => {
    const observers = [
      { id: "Elena", quat: quatFromYDeg(0) },
    ]
    const nodes = [
      { ...makeNode(0), id: "Elena" },
      { ...makeNode(10), id: "Other" },
    ]

    const results = evaluateMultiObserverPerceptions(observers, nodes, 0.016)

    // Elena should not appear in her own results
    const elenaEntries = results["Elena"]
    assert.ok(elenaEntries)
    const selfEntry = elenaEntries.find(r => r.node.id === "Elena")
    assert.strictEqual(selfEntry, undefined)

    // But other nodes should still be evaluated
    const otherEntry = elenaEntries.find(r => r.node.id === "Other")
    assert.ok(otherEntry)
    assert.notStrictEqual(otherEntry.state, SpinState.IDLE)
  })

  it("test_observer_dwell_accumulates", () => {
    const observers = [
      { id: "Cam", quat: quatFromYDeg(0) },
    ]
    const nodes = [
      { ...makeNode(5), id: "Target" },
    ]

    // First call
    const results1 = evaluateMultiObserverPerceptions(observers, nodes, 0.5)
    const dwell1 = results1["Cam"].find(r => r.node.id === "Target").dwellTime

    // Second call — dwell should accumulate on same node
    const results2 = evaluateMultiObserverPerceptions(observers, nodes, 0.5)
    const dwell2 = results2["Cam"].find(r => r.node.id === "Target").dwellTime

    assert.ok(dwell2 > dwell1, `dwell should accumulate: ${dwell2} > ${dwell1}`)
    assert.ok(Math.abs(dwell2 - 1.0) < EPS)
  })
})
