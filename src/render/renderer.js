/**
 * WebGL renderer with projection/view matrix support and SpinState-based rendering.
 *
 * Renders PlazaNodes onto a sphere around the camera. Visibility and opacity
 * are driven by each node's SpinState (idle → invisible, perceived → faint,
 * focused → full, activated → bright).
 *
 * Camera stays at origin; orientation is set from the head quaternion.
 *
 * @module render/renderer
 */

import { createGLContext } from "../core/GLContext.js"
import { SpinState } from "../core/perception.js"
import { quatConjugate } from "../core/spinstep.js"
import { orientationToPosition } from "../core/node.js"

// ---------------------------------------------------------------------------
// Matrix helpers (column-major for WebGL)
// ---------------------------------------------------------------------------

/**
 * Create a perspective projection matrix.
 * @param {number} fovRad - vertical field of view in radians
 * @param {number} aspect - width / height
 * @param {number} near
 * @param {number} far
 * @returns {Float32Array} 4x4 column-major matrix
 */
export function perspectiveMatrix(fovRad, aspect, near, far) {
  const f = 1.0 / Math.tan(fovRad / 2)
  const nf = 1.0 / (near - far)
  // prettier-ignore
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ])
}

/**
 * Create a view matrix from a head quaternion.
 * Camera at origin, orientation = inverse of head quaternion.
 * @param {number[]} headQuat - [x, y, z, w]
 * @returns {Float32Array} 4x4 column-major view matrix
 */
export function viewMatrixFromQuat(headQuat) {
  // View matrix = inverse of camera transform
  // Camera rotation = headQuat, so view rotation = conjugate
  const q = quatConjugate(headQuat)

  // Quaternion to rotation matrix (column-major)
  const [x, y, z, w] = q
  const x2 = x + x, y2 = y + y, z2 = z + z
  const xx = x * x2, xy = x * y2, xz = x * z2
  const yy = y * y2, yz = y * z2, zz = z * z2
  const wx = w * x2, wy = w * y2, wz = w * z2

  // prettier-ignore
  return new Float32Array([
    1 - (yy + zz), xy + wz, xz - wy, 0,
    xy - wz, 1 - (xx + zz), yz + wx, 0,
    xz + wy, yz - wx, 1 - (xx + yy), 0,
    0, 0, 0, 1,
  ])
}

/**
 * Multiply two 4x4 matrices (column-major).
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @returns {Float32Array}
 */
export function mat4Multiply(a, b) {
  const out = new Float32Array(16)
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      let sum = 0
      for (let k = 0; k < 4; k++) {
        sum += a[k * 4 + row] * b[col * 4 + k]
      }
      out[col * 4 + row] = sum
    }
  }
  return out
}

// ---------------------------------------------------------------------------
// Geometry data for entity types
// ---------------------------------------------------------------------------

/** Box vertices (36 verts = 12 triangles, CCW winding) */
function makeBoxVertices(sx = 1, sy = 1, sz = 1) {
  const hx = sx / 2, hy = sy / 2, hz = sz / 2
  // prettier-ignore
  return new Float32Array([
    // Front face
    -hx, -hy,  hz,   hx, -hy,  hz,   hx,  hy,  hz,
    -hx, -hy,  hz,   hx,  hy,  hz,  -hx,  hy,  hz,
    // Back face
    -hx, -hy, -hz,  -hx,  hy, -hz,   hx,  hy, -hz,
    -hx, -hy, -hz,   hx,  hy, -hz,   hx, -hy, -hz,
    // Top face
    -hx,  hy, -hz,  -hx,  hy,  hz,   hx,  hy,  hz,
    -hx,  hy, -hz,   hx,  hy,  hz,   hx,  hy, -hz,
    // Bottom face
    -hx, -hy, -hz,   hx, -hy, -hz,   hx, -hy,  hz,
    -hx, -hy, -hz,   hx, -hy,  hz,  -hx, -hy,  hz,
    // Right face
     hx, -hy, -hz,   hx,  hy, -hz,   hx,  hy,  hz,
     hx, -hy, -hz,   hx,  hy,  hz,   hx, -hy,  hz,
    // Left face
    -hx, -hy, -hz,  -hx, -hy,  hz,  -hx,  hy,  hz,
    -hx, -hy, -hz,  -hx,  hy,  hz,  -hx,  hy, -hz,
  ])
}

/** Sphere vertices (UV sphere) */
function makeSphereVertices(radius = 0.5, rings = 12, segments = 16) {
  const verts = []
  for (let r = 0; r < rings; r++) {
    const theta1 = (r / rings) * Math.PI
    const theta2 = ((r + 1) / rings) * Math.PI
    for (let s = 0; s < segments; s++) {
      const phi1 = (s / segments) * 2 * Math.PI
      const phi2 = ((s + 1) / segments) * 2 * Math.PI

      const p1 = spherePoint(radius, theta1, phi1)
      const p2 = spherePoint(radius, theta2, phi1)
      const p3 = spherePoint(radius, theta2, phi2)
      const p4 = spherePoint(radius, theta1, phi2)

      verts.push(...p1, ...p2, ...p3)
      verts.push(...p1, ...p3, ...p4)
    }
  }
  return new Float32Array(verts)
}

function spherePoint(r, theta, phi) {
  return [
    r * Math.sin(theta) * Math.cos(phi),
    r * Math.cos(theta),
    r * Math.sin(theta) * Math.sin(phi),
  ]
}

/** Quad vertices for panels (2 triangles forming a plane) */
function makePanelVertices(w = 1.2, h = 0.8) {
  const hw = w / 2, hh = h / 2
  // prettier-ignore
  return new Float32Array([
    -hw, -hh, 0,   hw, -hh, 0,   hw,  hh, 0,
    -hw, -hh, 0,   hw,  hh, 0,  -hw,  hh, 0,
  ])
}

// ---------------------------------------------------------------------------
// PlazaRenderer
// ---------------------------------------------------------------------------

/**
 * @typedef {object} GPUObject
 * @property {WebGLBuffer} buffer
 * @property {number} vertexCount
 */

export class PlazaRenderer {
  /**
   * @param {HTMLCanvasElement} canvas
   */
  constructor(canvas) {
    this.canvas = canvas
    this.gl = createGLContext(canvas)
    this.program = null
    /** @type {Map<string, GPUObject>} node id → GPU buffer */
    this.gpuObjects = new Map()
    /** Uniform locations */
    this.uniforms = {}
    /** Attribute locations */
    this.attribs = {}
    /** @type {Float32Array} reusable model matrix */
    this._modelMatrix = new Float32Array(16)

    this._initShaders()
    this._initState()
  }

  _initShaders() {
    const gl = this.gl

    const vsSource = `
      attribute vec3 aPosition;
      uniform mat4 uMVP;
      void main() {
        gl_Position = uMVP * vec4(aPosition, 1.0);
      }
    `

    const fsSource = `
      precision mediump float;
      uniform vec3 uColor;
      uniform float uOpacity;
      void main() {
        gl_FragColor = vec4(uColor, uOpacity);
      }
    `

    const vs = this._compileShader(gl.VERTEX_SHADER, vsSource)
    const fs = this._compileShader(gl.FRAGMENT_SHADER, fsSource)

    this.program = gl.createProgram()
    gl.attachShader(this.program, vs)
    gl.attachShader(this.program, fs)
    gl.linkProgram(this.program)

    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      throw new Error("Program link error: " + gl.getProgramInfoLog(this.program))
    }

    this.uniforms.uMVP = gl.getUniformLocation(this.program, "uMVP")
    this.uniforms.uColor = gl.getUniformLocation(this.program, "uColor")
    this.uniforms.uOpacity = gl.getUniformLocation(this.program, "uOpacity")
    this.attribs = {
      aPosition: gl.getAttribLocation(this.program, "aPosition"),
    }
  }

  _compileShader(type, source) {
    const gl = this.gl
    const shader = gl.createShader(type)
    gl.shaderSource(shader, source)
    gl.compileShader(shader)
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error("Shader compile error: " + gl.getShaderInfoLog(shader))
    }
    return shader
  }

  _initState() {
    const gl = this.gl
    gl.enable(gl.DEPTH_TEST)
    gl.enable(gl.BLEND)
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)
    gl.clearColor(0.05, 0.05, 0.12, 1.0)  // dark blue-black
  }

  /**
   * Resize the canvas and viewport to match display size.
   */
  resize() {
    const canvas = this.canvas
    const dpr = window.devicePixelRatio || 1
    const w = Math.floor(canvas.clientWidth * dpr)
    const h = Math.floor(canvas.clientHeight * dpr)
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w
      canvas.height = h
      this.gl.viewport(0, 0, w, h)
    }
  }

  /**
   * Ensure GPU buffers exist for a node.
   * @param {object} node - PlazaNode
   */
  ensureGPUObject(node) {
    if (this.gpuObjects.has(node.id)) return

    const gl = this.gl
    let vertices

    switch (node.entityType) {
      case "npc":
        vertices = makeSphereVertices(0.4, 8, 12)
        break
      case "object":
        vertices = makeBoxVertices(0.6, 0.6, 0.6)
        break
      case "panel":
        vertices = makePanelVertices(1.4, 1.0)
        break
      case "audio":
        // Audio nodes: small marker sphere
        vertices = makeSphereVertices(0.15, 6, 8)
        break
      default:
        vertices = makeBoxVertices(0.3, 0.3, 0.3)
    }

    const buffer = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer)
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW)

    this.gpuObjects.set(node.id, {
      buffer,
      vertexCount: vertices.length / 3,
    })
  }

  /**
   * Render the full scene.
   * @param {object[]} nodes - array of PlazaNodes
   * @param {number[]} headQuat - [x, y, z, w]
   * @param {number} sphereRadius - distance of objects from camera
   */
  render(nodes, headQuat, sphereRadius = 5) {
    const gl = this.gl
    this.resize()

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
    gl.useProgram(this.program)

    const aspect = this.canvas.width / this.canvas.height
    const proj = perspectiveMatrix(75 * Math.PI / 180, aspect, 0.1, 100)
    const view = viewMatrixFromQuat(headQuat)
    const vp = mat4Multiply(proj, view)

    for (const node of nodes) {
      // Skip idle nodes (not rendered) and audio-only nodes without visual
      if (node.spinState === SpinState.IDLE) continue

      this.ensureGPUObject(node)
      const gpu = this.gpuObjects.get(node.id)
      if (!gpu) continue

      // Compute world position from orientation
      const pos = orientationToPosition(node.orientation, sphereRadius)

      // Model matrix (translation only — no rotation needed for basic shapes)
      const model = this._modelMatrix
      model.fill(0)
      model[0] = 1; model[5] = 1; model[10] = 1; model[15] = 1
      model[12] = pos[0]
      model[13] = pos[1]
      model[14] = pos[2]

      // Scale for activated state (pulse effect)
      if (node.spinState === SpinState.ACTIVATED) {
        const pulse = 1.0 + 0.1 * Math.sin(Date.now() * 0.005)
        model[0] = pulse; model[5] = pulse; model[10] = pulse
      }

      const mvp = mat4Multiply(vp, model)

      // Set uniforms
      gl.uniformMatrix4fv(this.uniforms.uMVP, false, mvp)

      const color = node.metadata.color || [0.7, 0.7, 0.7]
      const opacity = getOpacity(node.spinState)
      const finalColor = getColorForState(color, node.spinState)
      gl.uniform3fv(this.uniforms.uColor, finalColor)
      gl.uniform1f(this.uniforms.uOpacity, opacity)

      // Draw
      gl.bindBuffer(gl.ARRAY_BUFFER, gpu.buffer)
      gl.enableVertexAttribArray(this.attribs.aPosition)
      gl.vertexAttribPointer(this.attribs.aPosition, 3, gl.FLOAT, false, 0, 0)
      gl.drawArrays(gl.TRIANGLES, 0, gpu.vertexCount)
    }
  }

  /**
   * Clean up all GPU resources.
   */
  dispose() {
    const gl = this.gl
    for (const [, gpu] of this.gpuObjects) {
      gl.deleteBuffer(gpu.buffer)
    }
    this.gpuObjects.clear()
    if (this.program) {
      gl.deleteProgram(this.program)
      this.program = null
    }
  }
}

// ---------------------------------------------------------------------------
// State → visual mapping
// ---------------------------------------------------------------------------

/**
 * Get opacity for a SpinState.
 * @param {string} state
 * @returns {number}
 */
export function getOpacity(state) {
  switch (state) {
    case SpinState.IDLE: return 0.0
    case SpinState.PERCEIVED: return 0.3
    case SpinState.FOCUSED: return 1.0
    case SpinState.ACTIVATED: return 1.0
    default: return 0.0
  }
}

/**
 * Modify color based on state (add emissive glow for focused/activated).
 * @param {number[]} baseColor - [r, g, b]
 * @param {string} state
 * @returns {Float32Array}
 */
export function getColorForState(baseColor, state) {
  const [r, g, b] = baseColor
  switch (state) {
    case SpinState.PERCEIVED:
      // Desaturated (move toward gray)
      return new Float32Array([r * 0.5 + 0.25, g * 0.5 + 0.25, b * 0.5 + 0.25])
    case SpinState.FOCUSED:
      return new Float32Array([r, g, b])
    case SpinState.ACTIVATED:
      // Brightened
      return new Float32Array([
        Math.min(1.0, r * 1.3),
        Math.min(1.0, g * 1.3),
        Math.min(1.0, b * 1.3),
      ])
    default:
      return new Float32Array([r, g, b])
  }
}
