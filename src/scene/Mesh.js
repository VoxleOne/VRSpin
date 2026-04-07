import { Object3D } from "./Object3D.js"

export class Mesh extends Object3D {
  constructor(geometry, shader) {
    if (!geometry || !geometry.vertices) {
      throw new Error("Mesh requires a geometry with vertices")
    }
    if (!shader || typeof shader.use !== "function") {
      throw new Error("Mesh requires a shader with a use() method")
    }

    super()
    this.geometry = geometry
    this.shader = shader
    this.buffer = null
  }

  init(gl) {
    this.buffer = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer)
    gl.bufferData(gl.ARRAY_BUFFER, this.geometry.vertices, gl.STATIC_DRAW)
  }

  draw(gl, camera) {
    if (!this.buffer) this.init(gl)

    this.shader.use()

    const posLoc = gl.getAttribLocation(this.shader.program, "position")

    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer)
    gl.enableVertexAttribArray(posLoc)
    gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0)

    const vertexCount = this.geometry.vertices.length / 3
    gl.drawArrays(gl.TRIANGLES, 0, vertexCount)
  }

  dispose(gl) {
    if (this.buffer) {
      gl.deleteBuffer(this.buffer)
      this.buffer = null
    }
  }
}
