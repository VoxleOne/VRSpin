import { describe, it } from "node:test"
import assert from "node:assert/strict"

import {
  quatNormalize,
  quatMultiply,
  quatConjugate,
  quaternionDistance,
  isWithinAngle,
  quatRotateVec3,
  forwardFromQuat,
  quatFromYDeg,
  slerp,
  SpinNode,
  quaternionDepthIterate,
} from "./spinstep.js"

// Tolerance for floating point comparisons
const EPS = 1e-6

function assertClose(actual, expected, msg, tolerance = EPS) {
  assert.ok(
    Math.abs(actual - expected) < tolerance,
    `${msg}: expected ${expected}, got ${actual}`
  )
}

function assertVec3Close(actual, expected, msg) {
  for (let i = 0; i < 3; i++) {
    assertClose(actual[i], expected[i], `${msg}[${i}]`)
  }
}

describe("spinstep — quaternion math", () => {
  it("quatNormalize normalizes a quaternion", () => {
    const q = quatNormalize([0, 0, 0, 2])
    assertClose(q[3], 1.0, "w component")
    assertClose(q[0], 0.0, "x component")
  })

  it("quatNormalize handles zero quaternion", () => {
    const q = quatNormalize([0, 0, 0, 0])
    assert.deepStrictEqual(q, [0, 0, 0, 1])
  })

  it("quatMultiply identity * identity = identity", () => {
    const id = [0, 0, 0, 1]
    const result = quatMultiply(id, id)
    assertClose(result[3], 1.0, "w")
    assertClose(result[0], 0.0, "x")
  })

  it("quatConjugate negates xyz, keeps w", () => {
    const q = [0.1, 0.2, 0.3, 0.9]
    const c = quatConjugate(q)
    assertClose(c[0], -0.1, "x")
    assertClose(c[1], -0.2, "y")
    assertClose(c[2], -0.3, "z")
    assertClose(c[3], 0.9, "w")
  })

  it("quaternionDistance between identical quaternions is 0", () => {
    const q = [0, 0, 0, 1]
    assertClose(quaternionDistance(q, q), 0, "distance")
  })

  it("quaternionDistance between opposite quaternions is π", () => {
    // 180° rotation around Y: [0, 1, 0, 0]
    const q1 = [0, 0, 0, 1]
    const q2 = [0, 1, 0, 0]
    assertClose(quaternionDistance(q1, q2), Math.PI, "distance", 0.01)
  })

  it("quaternionDistance for 90° Y rotation is π/2", () => {
    const q1 = [0, 0, 0, 1]
    const q2 = quatFromYDeg(90)
    assertClose(quaternionDistance(q1, q2), Math.PI / 2, "distance", 0.01)
  })

  it("isWithinAngle returns true for close quaternions", () => {
    const q1 = [0, 0, 0, 1]
    const q2 = quatFromYDeg(10)
    assert.strictEqual(isWithinAngle(q1, q2, Math.PI / 6), true) // 30°
  })

  it("isWithinAngle returns false for far quaternions", () => {
    const q1 = [0, 0, 0, 1]
    const q2 = quatFromYDeg(70)
    assert.strictEqual(isWithinAngle(q1, q2, Math.PI / 3), false) // 60°
  })
})

describe("spinstep — vector rotation", () => {
  it("identity quaternion forward is [0, 0, -1]", () => {
    const fwd = forwardFromQuat([0, 0, 0, 1])
    assertVec3Close(fwd, [0, 0, -1], "forward")
  })

  it("90° Y rotation forward is [-1, 0, 0]", () => {
    const q = quatFromYDeg(90)
    const fwd = forwardFromQuat(q)
    assertVec3Close(fwd, [-1, 0, 0], "forward")
  })

  it("quatRotateVec3 with identity returns same vector", () => {
    const v = quatRotateVec3([0, 0, 0, 1], [1, 2, 3])
    assertVec3Close(v, [1, 2, 3], "rotated")
  })
})

describe("spinstep — quatFromYDeg", () => {
  it("0° returns identity", () => {
    const q = quatFromYDeg(0)
    assertClose(q[0], 0, "x")
    assertClose(q[1], 0, "y")
    assertClose(q[2], 0, "z")
    assertClose(q[3], 1, "w")
  })

  it("70° matches Python scipy result", () => {
    // Python: R.from_euler('y', 70, degrees=True).as_quat()
    // → [0, 0.57357644, 0, 0.81915204]
    const q = quatFromYDeg(70)
    assertClose(q[0], 0, "x")
    assertClose(q[1], 0.57357644, "y", 1e-4)
    assertClose(q[2], 0, "z")
    assertClose(q[3], 0.81915204, "w", 1e-4)
  })

  it("-70° matches Python scipy result", () => {
    const q = quatFromYDeg(-70)
    assertClose(q[0], 0, "x")
    assertClose(q[1], -0.57357644, "y", 1e-4)
    assertClose(q[3], 0.81915204, "w", 1e-4)
  })
})

describe("spinstep — slerp", () => {
  it("slerp at t=0 returns q1", () => {
    const q1 = [0, 0, 0, 1]
    const q2 = quatFromYDeg(90)
    const result = slerp(q1, q2, 0)
    assertClose(quaternionDistance(result, q1), 0, "distance", 0.01)
  })

  it("slerp at t=1 returns q2", () => {
    const q1 = [0, 0, 0, 1]
    const q2 = quatFromYDeg(90)
    const result = slerp(q1, q2, 1)
    assertClose(quaternionDistance(result, q2), 0, "distance", 0.01)
  })

  it("slerp at t=0.5 is halfway", () => {
    const q1 = [0, 0, 0, 1]
    const q2 = quatFromYDeg(90)
    const result = slerp(q1, q2, 0.5)
    const expected = quatFromYDeg(45)
    assertClose(quaternionDistance(result, expected), 0, "distance", 0.01)
  })
})

describe("spinstep — SpinNode", () => {
  it("creates a node with normalized orientation", () => {
    const node = new SpinNode("test", [0, 0, 0, 2])
    assertClose(node.orientation[3], 1.0, "w normalized")
    assert.strictEqual(node.name, "test")
    assert.deepStrictEqual(node.children, [])
  })

  it("accepts children", () => {
    const child = new SpinNode("child", [0, 0, 0, 1])
    const parent = new SpinNode("parent", [0, 0, 0, 1], [child])
    assert.strictEqual(parent.children.length, 1)
    assert.strictEqual(parent.children[0].name, "child")
  })
})

describe("spinstep — quaternionDepthIterate", () => {
  it("visits root node", () => {
    const root = new SpinNode("root", [0, 0, 0, 1])
    const visited = quaternionDepthIterate(root, [0, 0, 0, 1])
    assert.strictEqual(visited.length, 1)
    assert.strictEqual(visited[0].name, "root")
  })

  it("visits aligned children", () => {
    const child = new SpinNode("child", quatFromYDeg(5))
    const root = new SpinNode("root", [0, 0, 0, 1], [child])
    // Step with small angle, threshold should include child at 5°
    const visited = quaternionDepthIterate(root, quatFromYDeg(5), Math.PI / 6)
    const names = visited.map(n => n.name)
    assert.ok(names.includes("root"))
    assert.ok(names.includes("child"))
  })

  it("skips children outside threshold", () => {
    const child = new SpinNode("far_child", quatFromYDeg(90))
    const root = new SpinNode("root", [0, 0, 0, 1], [child])
    const visited = quaternionDepthIterate(root, quatFromYDeg(5), Math.PI / 18) // 10°
    const names = visited.map(n => n.name)
    assert.ok(names.includes("root"))
    assert.ok(!names.includes("far_child"))
  })
})
