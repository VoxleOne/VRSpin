export class Mesh {
  constructor(geometry, shader) {
    this.geometry = geometry
    this.shader = shader
    this.buffer = null
  }

  init(gl) {
    this.buffer = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer)
    gl.bufferData(gl.ARRAY_BUFFER, this.geometry.vertices, gl.STATIC_DRAW)
  }

  draw(gl) {
    if (!this.buffer) this.init(gl)

    this.shader.use()

    const posLoc = gl.getAttribLocation(this.shader.program, "position")

    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer)
    gl.enableVertexAttribArray(posLoc)
    gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0)

    gl.drawArrays(gl.TRIANGLES, 0, 3)
  }
}
