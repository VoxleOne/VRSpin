/**
 * WebGL renderer with Blinn-Phong lighting, materials, and fog.
 *
 * Renders PlazaNodes onto a sphere around the camera with per-entity
 * materials and SpinState-driven visibility. Includes a ground plane
 * for spatial anchoring and distance fog for atmosphere.
 *
 * Lighting: 2-light Blinn-Phong (key + fill) with hemisphere ambient,
 * per-entity specular/emissive/rim material, and linear distance fog.
 *
 * Camera stays at origin; orientation is set from the head quaternion.
 *
 * @module render/renderer
 */

import { createGLContext } from "../core/GLContext.js"
import { SpinState } from "../core/perception.js"
import { quatConjugate } from "../core/spinstep.js"
import { orientationToPosition } from "../core/node.js"
import {
  VERTEX_STRIDE, VERTEX_STRIDE_BYTES, POSITION_OFFSET, NORMAL_OFFSET,
  makeFountainGeometry, makeMarketStandGeometry, makeNPCGeometry,
  makeKnowledgePanelGeometry, makeAudioMarkerGeometry,
  makeBoxGeometry, makeGroundPlaneGeometry,
} from "./geometry.js"
import { getMaterial, getStateMaterial } from "./materials.js"

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
// Shader sources
// ---------------------------------------------------------------------------

const VERTEX_SHADER = `
  attribute vec3 aPosition;
  attribute vec3 aNormal;
  uniform mat4 uMVP;
  uniform mat4 uModel;
  varying vec3 vWorldPos;
  varying vec3 vNormal;
  void main() {
    vWorldPos = (uModel * vec4(aPosition, 1.0)).xyz;
    vNormal = aNormal;
    gl_Position = uMVP * vec4(aPosition, 1.0);
  }
`

const FRAGMENT_SHADER = `
  precision mediump float;

  // Lighting
  uniform vec3 uLightDir;
  uniform vec3 uLightColor;
  uniform vec3 uFillLightDir;
  uniform vec3 uFillLightColor;
  uniform vec3 uAmbientColor;

  // Material
  uniform vec3 uColor;
  uniform vec3 uSpecularColor;
  uniform float uShininess;
  uniform float uEmissive;
  uniform float uRimStrength;
  uniform float uOpacity;

  // Fog
  uniform vec3 uFogColor;
  uniform float uFogNear;
  uniform float uFogFar;

  varying vec3 vWorldPos;
  varying vec3 vNormal;

  void main() {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(-vWorldPos);

    // Key light (Blinn-Phong)
    float diff1 = max(dot(N, uLightDir), 0.0);
    vec3 H1 = normalize(uLightDir + V);
    float spec1 = pow(max(dot(N, H1), 0.0), uShininess);

    // Fill light
    float diff2 = max(dot(N, uFillLightDir), 0.0);
    vec3 H2 = normalize(uFillLightDir + V);
    float spec2 = pow(max(dot(N, H2), 0.0), uShininess) * 0.3;

    // Rim / fresnel light
    float rim = pow(1.0 - max(dot(N, V), 0.0), 3.0) * uRimStrength;

    // Combine
    vec3 ambient  = uAmbientColor * uColor;
    vec3 diffuse  = uColor * (diff1 * uLightColor + diff2 * uFillLightColor);
    vec3 specular = uSpecularColor * (spec1 * uLightColor + spec2 * uFillLightColor);
    vec3 emissive = uColor * uEmissive;
    vec3 rimCol   = vec3(0.4, 0.5, 0.7) * rim;

    vec3 finalColor = ambient + diffuse + specular + emissive + rimCol;

    // Distance fog (linear)
    float dist = length(vWorldPos);
    float fogFactor = clamp((uFogFar - dist) / (uFogFar - uFogNear), 0.0, 1.0);
    finalColor = mix(uFogColor, finalColor, fogFactor);

    gl_FragColor = vec4(finalColor, uOpacity);
  }
`

// ---------------------------------------------------------------------------
// Lighting & environment constants
// ---------------------------------------------------------------------------

function normalize3(v) {
  const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
  return [v[0] / len, v[1] / len, v[2] / len]
}

/** Key light — warm white from upper-right. */
const KEY_LIGHT_DIR = normalize3([1.0, 1.5, 0.5])
const KEY_LIGHT_COLOR = [1.0, 0.95, 0.85]

/** Fill light — cool blue from left-below. */
const FILL_LIGHT_DIR = normalize3([-0.5, -0.3, 0.8])
const FILL_LIGHT_COLOR = [0.35, 0.4, 0.55]

/** Ambient — dark blue-purple. */
const AMBIENT_COLOR = [0.12, 0.12, 0.18]

/** Fog — matches clear color for seamless distance fade. */
const FOG_COLOR = [0.05, 0.05, 0.12]
const FOG_NEAR = 3.0
const FOG_FAR = 12.0

/** Ground plane constants. */
const GROUND_Y = -1.5
const GROUND_COLOR = [0.08, 0.08, 0.14]

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
    /** @type {GPUObject|null} ground plane GPU data */
    this._groundGPU = null

    this._initShaders()
    this._initState()
    this._initGroundPlane()
  }

  _initShaders() {
    const gl = this.gl

    const vs = this._compileShader(gl.VERTEX_SHADER, VERTEX_SHADER)
    const fs = this._compileShader(gl.FRAGMENT_SHADER, FRAGMENT_SHADER)

    this.program = gl.createProgram()
    gl.attachShader(this.program, vs)
    gl.attachShader(this.program, fs)
    gl.linkProgram(this.program)

    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      throw new Error("Program link error: " + gl.getProgramInfoLog(this.program))
    }

    // Cache all uniform locations
    const uniformNames = [
      "uMVP", "uModel",
      "uLightDir", "uLightColor", "uFillLightDir", "uFillLightColor", "uAmbientColor",
      "uColor", "uSpecularColor", "uShininess", "uEmissive", "uRimStrength", "uOpacity",
      "uFogColor", "uFogNear", "uFogFar",
    ]
    for (const name of uniformNames) {
      this.uniforms[name] = gl.getUniformLocation(this.program, name)
    }

    this.attribs = {
      aPosition: gl.getAttribLocation(this.program, "aPosition"),
      aNormal: gl.getAttribLocation(this.program, "aNormal"),
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
    gl.clearColor(...FOG_COLOR, 1.0)
  }

  /**
   * Initialize the ground plane GPU buffer.
   */
  _initGroundPlane() {
    const gl = this.gl
    const geo = makeGroundPlaneGeometry(25)
    const buffer = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer)
    gl.bufferData(gl.ARRAY_BUFFER, geo.data, gl.STATIC_DRAW)
    this._groundGPU = { buffer, vertexCount: geo.vertexCount }
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
   * Ensure GPU buffers exist for a node. Uses entity-specific geometry.
   * @param {object} node - PlazaNode
   */
  ensureGPUObject(node) {
    if (this.gpuObjects.has(node.id)) return

    const gl = this.gl
    let geo

    // Entity-specific geometry selection
    if (node.id === "Fountain") {
      geo = makeFountainGeometry()
    } else if (node.id === "MarketStand") {
      geo = makeMarketStandGeometry()
    } else {
      switch (node.entityType) {
        case "npc":
          geo = makeNPCGeometry()
          break
        case "panel":
          geo = makeKnowledgePanelGeometry()
          break
        case "audio":
          geo = makeAudioMarkerGeometry()
          break
        case "object":
        default:
          geo = makeBoxGeometry(0.6, 0.6, 0.6)
          break
      }
    }

    const buffer = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer)
    gl.bufferData(gl.ARRAY_BUFFER, geo.data, gl.STATIC_DRAW)

    this.gpuObjects.set(node.id, {
      buffer,
      vertexCount: geo.vertexCount,
    })
  }

  /**
   * Set per-frame uniforms (lighting, fog) that don't change per entity.
   */
  _setFrameUniforms() {
    const gl = this.gl
    gl.uniform3fv(this.uniforms.uLightDir, KEY_LIGHT_DIR)
    gl.uniform3fv(this.uniforms.uLightColor, KEY_LIGHT_COLOR)
    gl.uniform3fv(this.uniforms.uFillLightDir, FILL_LIGHT_DIR)
    gl.uniform3fv(this.uniforms.uFillLightColor, FILL_LIGHT_COLOR)
    gl.uniform3fv(this.uniforms.uAmbientColor, AMBIENT_COLOR)
    gl.uniform3fv(this.uniforms.uFogColor, FOG_COLOR)
    gl.uniform1f(this.uniforms.uFogNear, FOG_NEAR)
    gl.uniform1f(this.uniforms.uFogFar, FOG_FAR)
  }

  /**
   * Bind and draw a single GPU object with interleaved pos+normal attributes.
   * @param {GPUObject} gpu
   */
  _drawGPUObject(gpu) {
    const gl = this.gl
    gl.bindBuffer(gl.ARRAY_BUFFER, gpu.buffer)
    gl.enableVertexAttribArray(this.attribs.aPosition)
    gl.vertexAttribPointer(
      this.attribs.aPosition, 3, gl.FLOAT, false,
      VERTEX_STRIDE_BYTES, POSITION_OFFSET,
    )
    if (this.attribs.aNormal >= 0) {
      gl.enableVertexAttribArray(this.attribs.aNormal)
      gl.vertexAttribPointer(
        this.attribs.aNormal, 3, gl.FLOAT, false,
        VERTEX_STRIDE_BYTES, NORMAL_OFFSET,
      )
    }
    gl.drawArrays(gl.TRIANGLES, 0, gpu.vertexCount)
  }

  /**
   * Set material uniforms for a given material definition.
   * @param {object} mat - { specularColor, shininess, emissive, rimStrength }
   */
  _setMaterialUniforms(mat) {
    const gl = this.gl
    gl.uniform3fv(this.uniforms.uSpecularColor, mat.specularColor)
    gl.uniform1f(this.uniforms.uShininess, mat.shininess)
    gl.uniform1f(this.uniforms.uEmissive, mat.emissive)
    gl.uniform1f(this.uniforms.uRimStrength, mat.rimStrength)
  }

  /**
   * Render the ground plane (always visible, provides spatial anchoring).
   * @param {Float32Array} vp - view-projection matrix
   */
  _renderGroundPlane(vp) {
    if (!this._groundGPU) return
    const gl = this.gl

    // Ground model matrix — translate to GROUND_Y
    const model = this._modelMatrix
    model.fill(0)
    model[0] = 1; model[5] = 1; model[10] = 1; model[15] = 1
    model[13] = GROUND_Y

    const mvp = mat4Multiply(vp, model)
    gl.uniformMatrix4fv(this.uniforms.uMVP, false, mvp)
    gl.uniformMatrix4fv(this.uniforms.uModel, false, model)

    // Ground material + color
    gl.uniform3fv(this.uniforms.uColor, GROUND_COLOR)
    gl.uniform1f(this.uniforms.uOpacity, 1.0)
    this._setMaterialUniforms({
      specularColor: [0.1, 0.1, 0.1],
      shininess: 4.0,
      emissive: 0.02,
      rimStrength: 0.0,
    })

    this._drawGPUObject(this._groundGPU)
  }

  /**
   * Render the full scene.
   * @param {object[]} nodes - array of PlazaNodes
   * @param {number[]} headQuat - [x, y, z, w]
   * @param {number} sphereRadius - default distance of objects from camera
   * @param {number} [time] - current time in ms (defaults to performance.now())
   */
  render(nodes, headQuat, sphereRadius = 5, time = performance.now()) {
    const gl = this.gl
    this.resize()

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
    gl.useProgram(this.program)

    const aspect = this.canvas.width / this.canvas.height
    const proj = perspectiveMatrix(75 * Math.PI / 180, aspect, 0.1, 100)
    const view = viewMatrixFromQuat(headQuat)
    const vp = mat4Multiply(proj, view)

    // Set per-frame lighting and fog uniforms
    this._setFrameUniforms()

    // Render ground plane (always visible)
    this._renderGroundPlane(vp)

    // Render scene entities
    for (const node of nodes) {
      // Skip idle nodes (not rendered)
      if (node.spinState === SpinState.IDLE) continue

      this.ensureGPUObject(node)
      const gpu = this.gpuObjects.get(node.id)
      if (!gpu) continue

      // Compute world position from orientation + per-entity render hints
      const radius = node.metadata.renderRadius || sphereRadius
      const yOffset = node.metadata.renderYOffset || 0
      const pos = orientationToPosition(node.orientation, radius)
      pos[1] += yOffset

      // Model matrix (translation + optional uniform scale)
      const model = this._modelMatrix
      model.fill(0)
      model[0] = 1; model[5] = 1; model[10] = 1; model[15] = 1
      model[12] = pos[0]
      model[13] = pos[1]
      model[14] = pos[2]

      // Scale for activated state (pulse effect)
      if (node.spinState === SpinState.ACTIVATED) {
        const pulse = 1.0 + 0.1 * Math.sin(time * 0.005)
        model[0] = pulse; model[5] = pulse; model[10] = pulse
      }

      const mvp = mat4Multiply(vp, model)
      gl.uniformMatrix4fv(this.uniforms.uMVP, false, mvp)
      gl.uniformMatrix4fv(this.uniforms.uModel, false, model)

      // Color + opacity from SpinState
      const color = node.metadata.color || [0.7, 0.7, 0.7]
      const opacity = getOpacity(node.spinState)
      const finalColor = getColorForState(color, node.spinState)
      gl.uniform3fv(this.uniforms.uColor, finalColor)
      gl.uniform1f(this.uniforms.uOpacity, opacity)

      // Material from entity type + SpinState modification
      const baseMat = getMaterial(node)
      const mat = getStateMaterial(baseMat, node.spinState)
      this._setMaterialUniforms(mat)

      // Draw
      this._drawGPUObject(gpu)
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
    if (this._groundGPU) {
      gl.deleteBuffer(this._groundGPU.buffer)
      this._groundGPU = null
    }
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
