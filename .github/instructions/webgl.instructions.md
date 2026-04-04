You are acting as the Repository Master Refactor

Your task is to refactor this repository by introducing a **lightweight WebGL-based rendering system**.

The goal is to:

- Add visualization capabilities
- Keep architecture modular
- Avoid disrupting existing code

------

# Rules of Engagement

- Do NOT modify existing business logic
- Integrate via **adapter/app layer only**
- Keep WebGL concerns isolated
- Build incrementally (render first, features later)

------

# Target Architecture

```
App Layer (integration)
        ↓
Scene System
        ↓
Renderer (WebGL abstraction)
        ↓
WebGL API
```

------

# Step 1 — Create Folder Structure

Inside your repo:

```
/src/
  core/
  renderer/
  scene/
  geometry/
  materials/
  utils/
  app/
shaders/
```

------

# Step 2 — Core Implementation

## `/src/core/GLContext.js`

```
export function createGLContext(canvas) {
  const gl = canvas.getContext("webgl2") || canvas.getContext("webgl")

  if (!gl) {
    throw new Error("WebGL not supported")
  }

  return gl
}
```

------

## `/src/core/Renderer.js`

```
export class Renderer {
  constructor(gl) {
    this.gl = gl
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
```

------

# Step 3 — Scene System

## `/src/scene/Scene.js`

```
export class Scene {
  constructor() {
    this.objects = []
  }

  add(obj) {
    this.objects.push(obj)
  }
}
```

------

## `/src/scene/Camera.js`

```
export class Camera {
  constructor() {
    this.projectionMatrix = mat4.create()
    this.viewMatrix = mat4.create()
  }
}
```

------

## `/src/scene/Object3D.js`

```
export class Object3D {
  constructor() {
    this.position = [0, 0, 0]
  }
}
```

------

# Step 4 — Geometry

## `/src/geometry/Geometry.js`

```
export class Geometry {
  constructor(vertices) {
    this.vertices = vertices
  }
}
```

------

## `/src/geometry/BoxGeometry.js`

```
import { Geometry } from "./Geometry.js"

export class BoxGeometry extends Geometry {
  constructor() {
    super(new Float32Array([
      -0.5, -0.5, 0,
       0.5, -0.5, 0,
       0.0,  0.5, 0
    ]))
  }
}
```

------

# Step 5 — Shader System

## `/src/renderer/ShaderProgram.js`

```
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
```

------

# Step 6 — Mesh (Renderable Object)

## `/src/scene/Mesh.js`

```
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
```

------

# Step 7 — Basic Shaders

## `/shaders/basic.vert.glsl`

```
attribute vec3 position;

void main() {
  gl_Position = vec4(position, 1.0);
}
```

------

## `/shaders/basic.frag.glsl`

```
precision mediump float;

void main() {
  gl_FragColor = vec4(1.0, 0.5, 0.2, 1.0);
}
```

------

# 🚀 Step 8 — App Integration

## `/src/app/main.js`

```
import { createGLContext } from "../core/GLContext.js"
import { Renderer } from "../core/Renderer.js"
import { Scene } from "../scene/Scene.js"
import { Mesh } from "../scene/Mesh.js"
import { BoxGeometry } from "../geometry/BoxGeometry.js"
import { ShaderProgram } from "../renderer/ShaderProgram.js"

import vs from "../../shaders/basic.vert.glsl"
import fs from "../../shaders/basic.frag.glsl"

const canvas = document.querySelector("canvas")
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
```

------

# Step 9 — Integration Strategy

To connect with our existing repository:

### Option A — Data Adapter

```
Existing Data → Adapter → Geometry → Mesh
```

### Option B — Direct Hook

- Use existing data structures
- Convert them into Float32Array buffers

------

# Step 10 — Validation Checklist

- Canvas renders without errors
- Triangle appears on screen
- No WebGL warnings in console
- Render loop is stable

------

# Next Iterations (Incremental)

Add features in this order:

1. Transform matrices
2. Camera projection
3. Colors/uniforms
4. Textures
5. Multiple objects
6. Basic lighting

------

# Notes

- This is intentionally **minimal**
- Designed for **clarity over abstraction**
- Expand only when needed

------

# Deliverables

After completing this guide, the repository will have:

- A working WebGL renderer
- Clean modular structure
- Safe integration point
- Foundation for future features

------

**End of Document**
