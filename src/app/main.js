import { createGLContext } from "../core/GLContext.js"
import { Renderer } from "../core/Renderer.js"
import { Scene } from "../scene/Scene.js"
import { Camera } from "../scene/Camera.js"
import { Mesh } from "../scene/Mesh.js"
import { TriangleGeometry } from "../geometry/TriangleGeometry.js"
import { ShaderProgram } from "../renderer/ShaderProgram.js"

async function loadShader(url) {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Failed to load shader: ${url}`)
  }
  return response.text()
}

async function main() {
  const canvas = document.querySelector("canvas")
  if (!canvas) {
    throw new Error("No <canvas> element found in the document")
  }

  canvas.width = canvas.clientWidth
  canvas.height = canvas.clientHeight

  if (canvas.width === 0 || canvas.height === 0) {
    throw new Error("Canvas has zero size — check CSS or layout")
  }

  const [vertexShaderSource, fragmentShaderSource] = await Promise.all([
    loadShader("shaders/basic.vert.glsl"),
    loadShader("shaders/basic.frag.glsl")
  ])

  const gl = createGLContext(canvas)
  const renderer = new Renderer(gl)
  renderer.setSize(canvas.width, canvas.height)

  window.addEventListener("resize", () => {
    canvas.width = canvas.clientWidth
    canvas.height = canvas.clientHeight
    renderer.setSize(canvas.width, canvas.height)
  })

  const scene = new Scene()
  const camera = new Camera()

  const shader = new ShaderProgram(gl, vertexShaderSource, fragmentShaderSource)
  const geometry = new TriangleGeometry()

  const mesh = new Mesh(geometry, shader)
  scene.add(mesh)

  function loop() {
    renderer.render(scene, camera)
    requestAnimationFrame(loop)
  }

  loop()
}

main()
