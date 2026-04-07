/**
 * Lightweight WebGL context mock for unit testing.
 * Provides stub implementations of the WebGL methods used by the rendering system.
 */
export function createMockGL() {
  const buffers = []
  const calls = []

  function record(name, args) {
    calls.push({ name, args: Array.from(args) })
  }

  return {
    ARRAY_BUFFER: 0x8892,
    STATIC_DRAW: 0x88E4,
    FLOAT: 0x1406,
    TRIANGLES: 0x0004,
    COLOR_BUFFER_BIT: 0x4000,
    DEPTH_BUFFER_BIT: 0x0100,
    VERTEX_SHADER: 0x8B31,
    FRAGMENT_SHADER: 0x8B30,
    COMPILE_STATUS: 0x8B81,
    LINK_STATUS: 0x8B82,

    createBuffer() {
      const buf = { id: buffers.length }
      buffers.push(buf)
      record("createBuffer", arguments)
      return buf
    },

    deleteBuffer(buf) {
      record("deleteBuffer", arguments)
    },

    bindBuffer(target, buffer) {
      record("bindBuffer", arguments)
    },

    bufferData(target, data, usage) {
      record("bufferData", arguments)
    },

    getAttribLocation(program, name) {
      record("getAttribLocation", arguments)
      return 0
    },

    enableVertexAttribArray(index) {
      record("enableVertexAttribArray", arguments)
    },

    vertexAttribPointer(index, size, type, normalized, stride, offset) {
      record("vertexAttribPointer", arguments)
    },

    drawArrays(mode, first, count) {
      record("drawArrays", arguments)
    },

    useProgram(program) {
      record("useProgram", arguments)
    },

    clearColor(r, g, b, a) {
      record("clearColor", arguments)
    },

    clear(mask) {
      record("clear", arguments)
    },

    viewport(x, y, w, h) {
      record("viewport", arguments)
    },

    createShader(type) {
      record("createShader", arguments)
      return { type }
    },

    shaderSource(shader, source) {
      record("shaderSource", arguments)
    },

    compileShader(shader) {
      record("compileShader", arguments)
    },

    getShaderParameter(shader, pname) {
      return true
    },

    getShaderInfoLog(shader) {
      return ""
    },

    deleteShader(shader) {
      record("deleteShader", arguments)
    },

    createProgram() {
      record("createProgram", arguments)
      return { id: "program" }
    },

    attachShader(program, shader) {
      record("attachShader", arguments)
    },

    linkProgram(program) {
      record("linkProgram", arguments)
    },

    getProgramParameter(program, pname) {
      return true
    },

    getProgramInfoLog(program) {
      return ""
    },

    deleteProgram(program) {
      record("deleteProgram", arguments)
    },

    /** Retrieve all recorded WebGL calls for assertions. */
    _getCalls() {
      return calls
    },

    /** Retrieve calls filtered by method name. */
    _getCallsByName(name) {
      return calls.filter(c => c.name === name)
    },

    /** Reset recorded calls. */
    _reset() {
      calls.length = 0
    }
  }
}
