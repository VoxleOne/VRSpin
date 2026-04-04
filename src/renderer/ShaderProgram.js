export class ShaderProgram {
  constructor(gl, vertexSrc, fragmentSrc) {
    this.gl = gl
    this.program = this.createProgram(vertexSrc, fragmentSrc)
  }

  createShader(type, source) {
    const gl = this.gl
    const shader = gl.createShader(type)

    gl.shaderSource(shader, source)
    gl.compileShader(shader)

    return shader
  }

  createProgram(vsSource, fsSource) {
    const gl = this.gl

    const vs = this.createShader(gl.VERTEX_SHADER, vsSource)
    const fs = this.createShader(gl.FRAGMENT_SHADER, fsSource)

    const program = gl.createProgram()
    gl.attachShader(program, vs)
    gl.attachShader(program, fs)
    gl.linkProgram(program)

    return program
  }

  use() {
    this.gl.useProgram(this.program)
  }
}
