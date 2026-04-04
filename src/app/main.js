import { createGLContext } from "../core/GLContext.js"
import { Renderer } from "../core/Renderer.js"
import { Scene } from "../scene/Scene.js"
import { Mesh } from "../scene/Mesh.js"
import { BoxGeometry } from "../geometry/BoxGeometry.js"
import { ShaderProgram } from "../renderer/ShaderProgram.js"

const vertexShaderSource = `
attribute vec3 position;

void main() {
  gl_Position = vec4(position, 1.0);
}
`

const fragmentShaderSource = `
precision mediump float;

void main() {
  gl_FragColor = vec4(1.0, 0.5, 0.2, 1.0);
}
`

const canvas = document.querySelector("canvas")
if (!canvas) {
  throw new Error("No <canvas> element found in the document")
}

canvas.width = canvas.clientWidth
canvas.height = canvas.clientHeight

const gl = createGLContext(canvas)
const renderer = new Renderer(gl)
renderer.setSize(canvas.width, canvas.height)

window.addEventListener("resize", () => {
  canvas.width = canvas.clientWidth
  canvas.height = canvas.clientHeight
  renderer.setSize(canvas.width, canvas.height)
})

const scene = new Scene()

const shader = new ShaderProgram(gl, vertexShaderSource, fragmentShaderSource)
const geometry = new BoxGeometry()

const mesh = new Mesh(geometry, shader)
scene.add(mesh)

function loop() {
  renderer.render(scene)
  requestAnimationFrame(loop)
}

loop()
