import { createGLContext } from "../core/GLContext.js"
import { Renderer } from "../core/Renderer.js"
import { Scene } from "../scene/Scene.js"
import { Mesh } from "../scene/Mesh.js"
import { BoxGeometry } from "../geometry/BoxGeometry.js"
import { ShaderProgram } from "../renderer/ShaderProgram.js"

import vs from "../../shaders/basic.vert.glsl"
import fs from "../../shaders/basic.frag.glsl"

const canvas = document.querySelector("canvas")
if (!canvas) {
  throw new Error("No <canvas> element found in the document")
}
const gl = createGLContext(canvas)

const renderer = new Renderer(gl)
const scene = new Scene()

const shader = new ShaderProgram(gl, vs, fs)
const geometry = new BoxGeometry()

const mesh = new Mesh(geometry, shader)
scene.add(mesh)

function loop() {
  renderer.render(scene)
  requestAnimationFrame(loop)
}

loop()
