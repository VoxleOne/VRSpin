import { describe, it } from "node:test"
import assert from "node:assert/strict"

import {
  perspectiveMatrix,
  viewMatrixFromQuat,
  mat4Multiply,
  getOpacity,
  getColorForState,
} from "./renderer.js"

import { SpinState } from "../core/perception.js"
import { quatFromYDeg } from "../core/spinstep.js"

const EPS = 1e-5

describe("renderer — perspectiveMatrix", () => {
  it("returns a 16-element Float32Array", () => {
    const m = perspectiveMatrix(Math.PI / 4, 1.5, 0.1, 100)
    assert.ok(m instanceof Float32Array)
    assert.strictEqual(m.length, 16)
  })

  it("has non-zero diagonal elements", () => {
    const m = perspectiveMatrix(Math.PI / 4, 1.5, 0.1, 100)
    assert.ok(Math.abs(m[0]) > 0)  // m[0][0]
    assert.ok(Math.abs(m[5]) > 0)  // m[1][1]
    assert.ok(Math.abs(m[10]) > 0) // m[2][2]
  })

  it("has -1 in perspective division slot", () => {
    const m = perspectiveMatrix(Math.PI / 4, 1.5, 0.1, 100)
    assert.ok(Math.abs(m[11] - (-1)) < EPS) // m[2][3] = -1
  })
})

describe("renderer — viewMatrixFromQuat", () => {
  it("identity quaternion produces identity-like matrix", () => {
    const m = viewMatrixFromQuat([0, 0, 0, 1])
    // Identity rotation → identity matrix
    assert.ok(Math.abs(m[0] - 1) < EPS)
    assert.ok(Math.abs(m[5] - 1) < EPS)
    assert.ok(Math.abs(m[10] - 1) < EPS)
    assert.ok(Math.abs(m[15] - 1) < EPS)
  })

  it("returns a 16-element Float32Array", () => {
    const m = viewMatrixFromQuat(quatFromYDeg(45))
    assert.ok(m instanceof Float32Array)
    assert.strictEqual(m.length, 16)
  })

  it("non-identity quaternion produces non-identity matrix", () => {
    const identity = viewMatrixFromQuat([0, 0, 0, 1])
    const rotated = viewMatrixFromQuat(quatFromYDeg(45))
    // At least some elements should differ
    let differs = false
    for (let i = 0; i < 16; i++) {
      if (Math.abs(identity[i] - rotated[i]) > EPS) {
        differs = true
        break
      }
    }
    assert.ok(differs)
  })
})

describe("renderer — mat4Multiply", () => {
  it("identity * identity = identity", () => {
    // prettier-ignore
    const I = new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ])
    const result = mat4Multiply(I, I)
    for (let i = 0; i < 16; i++) {
      assert.ok(Math.abs(result[i] - I[i]) < EPS, `element ${i}`)
    }
  })

  it("produces correct dimensions", () => {
    const a = perspectiveMatrix(Math.PI / 4, 1, 0.1, 100)
    const b = viewMatrixFromQuat([0, 0, 0, 1])
    const result = mat4Multiply(a, b)
    assert.ok(result instanceof Float32Array)
    assert.strictEqual(result.length, 16)
  })
})

describe("renderer — getOpacity", () => {
  it("idle → 0", () => {
    assert.strictEqual(getOpacity(SpinState.IDLE), 0.0)
  })

  it("perceived → 0.3", () => {
    assert.strictEqual(getOpacity(SpinState.PERCEIVED), 0.3)
  })

  it("focused → 1.0", () => {
    assert.strictEqual(getOpacity(SpinState.FOCUSED), 1.0)
  })

  it("activated → 1.0", () => {
    assert.strictEqual(getOpacity(SpinState.ACTIVATED), 1.0)
  })
})

describe("renderer — getColorForState", () => {
  it("perceived desaturates color", () => {
    const color = getColorForState([1, 0, 0], SpinState.PERCEIVED)
    // Desaturated: 1*0.5+0.25 = 0.75, 0*0.5+0.25 = 0.25, 0*0.5+0.25 = 0.25
    assert.ok(Math.abs(color[0] - 0.75) < EPS)
    assert.ok(Math.abs(color[1] - 0.25) < EPS)
    assert.ok(Math.abs(color[2] - 0.25) < EPS)
  })

  it("focused returns base color", () => {
    const color = getColorForState([0.5, 0.6, 0.7], SpinState.FOCUSED)
    assert.ok(Math.abs(color[0] - 0.5) < EPS)
    assert.ok(Math.abs(color[1] - 0.6) < EPS)
    assert.ok(Math.abs(color[2] - 0.7) < EPS)
  })

  it("activated brightens color", () => {
    const color = getColorForState([0.5, 0.5, 0.5], SpinState.ACTIVATED)
    assert.ok(color[0] > 0.5)
    assert.ok(color[1] > 0.5)
    assert.ok(color[2] > 0.5)
  })

  it("activated clamps to 1.0", () => {
    const color = getColorForState([1.0, 1.0, 1.0], SpinState.ACTIVATED)
    assert.ok(color[0] <= 1.0)
    assert.ok(color[1] <= 1.0)
    assert.ok(color[2] <= 1.0)
  })
})
