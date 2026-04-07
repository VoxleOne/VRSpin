export class Renderer {
  constructor(gl) {
    this.gl = gl
    gl.enable(gl.DEPTH_TEST)
  }

  setSize(width, height) {
    this.gl.viewport(0, 0, width, height)
  }

  clear() {
    const gl = this.gl
    gl.clearColor(0, 0, 0, 1)
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
  }

  render(scene, camera) {
    this.clear()

    scene.objects.forEach(obj => {
      obj.draw(this.gl, camera)
    })
  }
}
