import { describe, it, beforeEach } from "node:test"
import assert from "node:assert/strict"

import { Mesh } from "./Mesh.js"
import { createMockGL } from "../utils/gl-mock.js"

function makeGeometry(vertices) {
  return { vertices: vertices || new Float32Array([-0.5, -0.5, 0, 0.5, -0.5, 0, 0.0, 0.5, 0]) }
}

function makeShader() {
  return { program: { id: "test-program" }, use() {} }
}

describe("Mesh", () => {
  let gl

  beforeEach(() => {
    gl = createMockGL()
  })

  // ─── Constructor ───────────────────────────────────────────────

  describe("constructor", () => {
    it("stores geometry and shader references", () => {
      const geometry = makeGeometry()
      const shader = makeShader()
      const mesh = new Mesh(geometry, shader)

      assert.strictEqual(mesh.geometry, geometry)
      assert.strictEqual(mesh.shader, shader)
    })

    it("initializes buffer to null", () => {
      const mesh = new Mesh(makeGeometry(), makeShader())
      assert.strictEqual(mesh.buffer, null)
    })

    it("throws when geometry is null", () => {
      assert.throws(() => new Mesh(null, makeShader()), {
        message: /geometry/i
      })
    })

    it("throws when geometry has no vertices", () => {
      assert.throws(() => new Mesh({}, makeShader()), {
        message: /geometry/i
      })
    })

    it("throws when shader is null", () => {
      assert.throws(() => new Mesh(makeGeometry(), null), {
        message: /shader/i
      })
    })

    it("throws when shader has no use method", () => {
      assert.throws(() => new Mesh(makeGeometry(), { program: {} }), {
        message: /shader/i
      })
    })
  })

  // ─── init ──────────────────────────────────────────────────────

  describe("init", () => {
    it("creates a WebGL buffer", () => {
      const mesh = new Mesh(makeGeometry(), makeShader())
      mesh.init(gl)

      const createCalls = gl._getCallsByName("createBuffer")
      assert.strictEqual(createCalls.length, 1)
    })

    it("binds the buffer to ARRAY_BUFFER", () => {
      const mesh = new Mesh(makeGeometry(), makeShader())
      mesh.init(gl)

      const bindCalls = gl._getCallsByName("bindBuffer")
      assert.strictEqual(bindCalls.length, 1)
      assert.strictEqual(bindCalls[0].args[0], gl.ARRAY_BUFFER)
    })

    it("uploads vertex data with STATIC_DRAW", () => {
      const geometry = makeGeometry()
      const mesh = new Mesh(geometry, makeShader())
      mesh.init(gl)

      const dataCalls = gl._getCallsByName("bufferData")
      assert.strictEqual(dataCalls.length, 1)
      assert.strictEqual(dataCalls[0].args[0], gl.ARRAY_BUFFER)
      assert.strictEqual(dataCalls[0].args[1], geometry.vertices)
      assert.strictEqual(dataCalls[0].args[2], gl.STATIC_DRAW)
    })

    it("sets buffer property to the created buffer object", () => {
      const mesh = new Mesh(makeGeometry(), makeShader())
      assert.strictEqual(mesh.buffer, null)

      mesh.init(gl)
      assert.notStrictEqual(mesh.buffer, null)
    })
  })

  // ─── draw ──────────────────────────────────────────────────────

  describe("draw", () => {
    it("auto-initializes buffer on first draw", () => {
      const mesh = new Mesh(makeGeometry(), makeShader())
      assert.strictEqual(mesh.buffer, null)

      mesh.draw(gl)
      assert.notStrictEqual(mesh.buffer, null)
    })

    it("does not re-create buffer on subsequent draws", () => {
      const mesh = new Mesh(makeGeometry(), makeShader())
      mesh.draw(gl)
      mesh.draw(gl)

      const createCalls = gl._getCallsByName("createBuffer")
      assert.strictEqual(createCalls.length, 1)
    })

    it("activates the shader program", () => {
      const shader = makeShader()
      let useCalled = false
      shader.use = () => { useCalled = true }

      const mesh = new Mesh(makeGeometry(), shader)
      mesh.draw(gl)

      assert.strictEqual(useCalled, true)
    })

    it("looks up the 'position' attribute location", () => {
      const mesh = new Mesh(makeGeometry(), makeShader())
      mesh.draw(gl)

      const attrCalls = gl._getCallsByName("getAttribLocation")
      assert.ok(attrCalls.length >= 1)
      assert.strictEqual(attrCalls[0].args[1], "position")
    })

    it("enables and configures vertex attribute pointer", () => {
      const mesh = new Mesh(makeGeometry(), makeShader())
      mesh.draw(gl)

      const enableCalls = gl._getCallsByName("enableVertexAttribArray")
      assert.strictEqual(enableCalls.length, 1)

      const ptrCalls = gl._getCallsByName("vertexAttribPointer")
      assert.strictEqual(ptrCalls.length, 1)
      assert.strictEqual(ptrCalls[0].args[1], 3)        // size (vec3)
      assert.strictEqual(ptrCalls[0].args[2], gl.FLOAT)  // type
      assert.strictEqual(ptrCalls[0].args[3], false)      // normalized
      assert.strictEqual(ptrCalls[0].args[4], 0)          // stride
      assert.strictEqual(ptrCalls[0].args[5], 0)          // offset
    })

    it("issues drawArrays with TRIANGLES and correct vertex count", () => {
      // 3 vertices (9 floats / 3 components)
      const mesh = new Mesh(makeGeometry(), makeShader())
      mesh.draw(gl)

      const drawCalls = gl._getCallsByName("drawArrays")
      assert.strictEqual(drawCalls.length, 1)
      assert.strictEqual(drawCalls[0].args[0], gl.TRIANGLES)
      assert.strictEqual(drawCalls[0].args[1], 0)
      assert.strictEqual(drawCalls[0].args[2], 3)
    })

    it("computes vertex count dynamically from geometry", () => {
      // 6 vertices (18 floats / 3 components)
      const vertices = new Float32Array([
        -1, -1, 0,  1, -1, 0,  0, 1, 0,
        -1, -1, 0,  1, -1, 0,  0, 1, 0
      ])
      const mesh = new Mesh(makeGeometry(vertices), makeShader())
      mesh.draw(gl)

      const drawCalls = gl._getCallsByName("drawArrays")
      assert.strictEqual(drawCalls[0].args[2], 6)
    })
  })

  // ─── dispose ───────────────────────────────────────────────────

  describe("dispose", () => {
    it("deletes the buffer and resets to null", () => {
      const mesh = new Mesh(makeGeometry(), makeShader())
      mesh.init(gl)

      assert.notStrictEqual(mesh.buffer, null)
      mesh.dispose(gl)
      assert.strictEqual(mesh.buffer, null)

      const deleteCalls = gl._getCallsByName("deleteBuffer")
      assert.strictEqual(deleteCalls.length, 1)
    })

    it("is safe to call when buffer is already null", () => {
      const mesh = new Mesh(makeGeometry(), makeShader())
      mesh.dispose(gl) // should not throw

      const deleteCalls = gl._getCallsByName("deleteBuffer")
      assert.strictEqual(deleteCalls.length, 0)
    })

    it("allows re-initialization after dispose", () => {
      const mesh = new Mesh(makeGeometry(), makeShader())
      mesh.init(gl)
      mesh.dispose(gl)
      assert.strictEqual(mesh.buffer, null)

      mesh.draw(gl) // re-initializes buffer
      assert.notStrictEqual(mesh.buffer, null)

      const createCalls = gl._getCallsByName("createBuffer")
      assert.strictEqual(createCalls.length, 2)
    })
  })
})
